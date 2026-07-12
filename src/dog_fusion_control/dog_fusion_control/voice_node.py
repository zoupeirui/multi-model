#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
voice_node.py
语音识别节点（空格控制 + 队列解耦 + 稳定重采样 + JSON 输出）

优化目标：
1. 保留当前高准确率主链路：队列解耦 + 48k->16k 重采样 + 无硬静音裁剪
2. 增加工程稳健性：绝对路径、热词 UTF-8 检查、坏热词自动禁用
3. 降低卡顿风险：有界队列 + 回调非阻塞 + overflow 限频告警
4. 输出更适合融合：JSON 结构化消息 + 参数类命令
5. 命令匹配更稳：重复压缩、尾音裁剪、轻量近音纠错
"""

import sys
import re
import os
import json
import time
import signal
import termios
import tty
import queue
import threading
import traceback
from pathlib import Path

import numpy as np

signal.signal(signal.SIGPIPE, signal.SIG_IGN)

try:
    from scipy.signal import resample_poly
    _HAS_SCIPY = True
except Exception:
    _HAS_SCIPY = False

import sherpa_onnx
import sounddevice as sd
import rclpy
from rclpy.node import Node
from std_msgs.msg import String

try:
    from ament_index_python.packages import get_package_share_directory
    _HAS_AMENT_INDEX = True
except Exception:
    _HAS_AMENT_INDEX = False


# ======================== 基本配置 ========================
PACKAGE_NAME = "dog_fusion_control"

MIC_SAMPLE_RATE = 48000
MIC_DEVICE_ID = 11

RECOGNIZER_SAMPLE_RATE = 16000
FEATURE_DIM = 80

NUM_THREADS = 2
PROVIDER = "cpu"
DECODING_METHOD = "modified_beam_search"
MAX_ACTIVE_PATHS = 2

HOTWORDS_SCORE = 1.5
RULE2_MIN_TRAILING = 0.40
RULE3_MIN_UTTERANCE = 6.0

FINAL_TAIL_SILENCE_SEC = 0.45

AUDIO_BLOCKSIZE = 4800          # 100 ms
AUDIO_LATENCY = "high"
AUDIO_QUEUE_MAXSIZE = 24

# 这里只做“极轻”的门限，避免纯静音时白白解码；不做硬切断
PRE_RMS_GATE = 0.0020

PARTIAL_PRINT_MIN_INTERVAL = 0.20
OVERFLOW_WARN_INTERVAL_SEC = 2.0
QUEUE_FULL_WARN_INTERVAL_SEC = 2.0

ALLOW_MULTI_COMMANDS = True
COMMAND_PUBLISH_INTERVAL = 0.7

PUBLISH_RAW_TEXT = True
RAW_TEXT_TOPIC = "/voice_text"

# 只允许短语段做 fuzzy，避免自然句误触发
MAX_FUZZY_SEGMENT_LEN = 6

CONTROL_SEMANTICS_CONTINUOUS = "CONTINUOUS"
CONTROL_SEMANTICS_DISCRETE = "DISCRETE"



# ======================== 全局打印锁 ========================
_print_lock = threading.Lock()

def safe_print(msg="", end="\n", flush=True):
    with _print_lock:
        try:
            print(msg, end=end, flush=flush)
        except (BrokenPipeError, OSError):
            pass

def safe_write(msg):
    with _print_lock:
        try:
            sys.stdout.write(msg)
            sys.stdout.flush()
        except (BrokenPipeError, OSError):
            pass


# ======================== 路径解析 ========================
def _candidate_roots():
    roots = []
    cwd = Path.cwd().resolve()
    script_dir = Path(__file__).resolve().parent

    roots.append(cwd)
    roots.append(script_dir)
    roots.extend(script_dir.parents)

    env_root = os.environ.get("VOICE_NODE_ASSET_DIR", "").strip()
    if env_root:
        roots.insert(0, Path(env_root).resolve())

    if _HAS_AMENT_INDEX:
        try:
            share_dir = Path(get_package_share_directory(PACKAGE_NAME)).resolve()
            roots.insert(0, share_dir)
            roots.append(share_dir.parent)
        except Exception:
            pass

    uniq = []
    seen = set()
    for p in roots:
        s = str(p)
        if s not in seen:
            uniq.append(p)
            seen.add(s)
    return uniq

def resolve_model_dir():
    for root in _candidate_roots():
        cand = root / "model_zipformer"
        if cand.is_dir():
            return cand.resolve()
    raise FileNotFoundError(
        "未找到 model_zipformer 目录。"
        "请确认模型目录已安装，或设置环境变量 VOICE_NODE_ASSET_DIR。"
    )

def resolve_hotwords_file():
    for root in _candidate_roots():
        cand = root / "hotwords.txt"
        if cand.is_file():
            return cand.resolve()
    return (Path.cwd() / "hotwords.txt").resolve()


# ======================== 音频处理 ========================
_RESAMPLE_UP = 1
_RESAMPLE_DOWN = 3  # 48k -> 16k

def downsample_48k_to_16k(audio: np.ndarray) -> np.ndarray:
    if _HAS_SCIPY:
        return resample_poly(audio, _RESAMPLE_UP, _RESAMPLE_DOWN).astype(np.float32)
    # 回退：仍然能跑，但准确率会下降
    return audio[::_RESAMPLE_DOWN].copy().astype(np.float32)


# ======================== 文本清洗 / 归一化 ========================
ASCII_NOISE_RE = re.compile(r"[A-Za-z0-9_]+")
PUNCT_RE = re.compile(r"[^\u4e00-\u9fffA-Za-z0-9，,。.!！？?；;、\s_-]")
MULTI_SPACE_RE = re.compile(r"\s+")
DUP_CHAR_RE = re.compile(r"(.)\1{1,}")
DUP_PHRASE_RE = re.compile(r"(.{2,4})\1{1,}")
TRAILING_FILLER_RE = re.compile(r"[啊呀呃嗯哦喔唔呜哈啦呢吧嘛]+$")
TRAILING_PARTICLE_RE = re.compile(r"[你呢吧吗呀啊啦嘛哈哦喔唔呜呃嗯]+$")

def clean_text(text: str) -> str:
    text = text.strip()
    text = PUNCT_RE.sub("", text)
    text = MULTI_SPACE_RE.sub("", text)
    text = text.replace("，", ",").replace("。", ",").replace("！", ",").replace("?", ",").replace("？", ",")
    text = text.replace("；", ",").replace(";", ",").replace("、", ",").replace(".", ",").replace("!", ",")
    text = re.sub(r",+", ",", text).strip(",")
    return text

def normalize_text(text: str) -> str:
    text = clean_text(text)
    text = ASCII_NOISE_RE.sub("", text)

    # 先做口令归一
    replacements = [
        ("打个招呼", "打招呼"),
        ("问个好", "你好"),
        ("跳个舞", "跳舞"),
        ("向前走", "前进"),
        ("往前走", "前进"),
        ("向后退", "后退"),
        ("往后退", "后退"),
        ("停下", "停止"),
        ("停住", "停止"),
        ("开机", "启动"),
        ("关闭", "关机"),
        ("全系统休眠", "关机"),
        ("系统休眠", "关机"),
    ]
    for a, b in replacements:
        text = text.replace(a, b)

    # 压缩重复
    text = DUP_PHRASE_RE.sub(r"\1", text)
    text = DUP_CHAR_RE.sub(r"\1", text)

    # 清尾巴
    text = TRAILING_FILLER_RE.sub("", text)

    text = re.sub(r",+", ",", text).strip(",")
    return text

def strip_trailing_particles(seg: str) -> str:
    if len(seg) <= 2:
        return seg
    stripped = TRAILING_PARTICLE_RE.sub("", seg)
    if len(stripped) >= 2:
        return stripped
    return seg

def canonicalize_segment(seg: str) -> str:
    seg = normalize_text(seg)
    seg = seg.replace(",", "").strip()
    seg = strip_trailing_particles(seg)
    return seg


# ======================== 命令定义 ========================
COMMAND_SPECS = {
    "HELLO": {
        "type": "DISCRETE",
        "canonical": "打招呼",
        "aliases": ["打招呼", "你好"],
        "fuzzy_patterns": [
            re.compile(r"^打[招昭找照朝]呼$"),
            re.compile(r"^你[好号浩郝]$"),
        ],
    },
    "FORWARD": {
        "type": "DIRECTION",
        "canonical": "前进",
        "aliases": ["前进", "向前"],
        "fuzzy_patterns": [
            re.compile(r"^(向)?[前钱千浅签潜迁渐见间尖乾]{1,2}[进近劲尽晋静镜敬竟径净]{1,2}$"),
        ],
    },
    "BACKWARD": {
        "type": "DIRECTION",
        "canonical": "后退",
        "aliases": ["后退", "向后"],
        "fuzzy_patterns": [
            re.compile(r"^(向)?[后侯厚候]{1,2}[退腿对兑褪]{1,2}$"),
        ],
    },
    "LEFT_MOVE": {
        "type": "DIRECTION",
        "canonical": "左移",
        "aliases": ["左移"],
        "fuzzy_patterns": [
            re.compile(r"^[左佐坐作]{1,2}[移怡宜疑夷仪]{1,2}$"),
        ],
    },
    "RIGHT_MOVE": {
        "type": "DIRECTION",
        "canonical": "右移",
        "aliases": ["右移"],
        "fuzzy_patterns": [
            re.compile(r"^[右又佑幼]{1,2}[移怡宜疑夷仪]{1,2}$"),
        ],
    },
    "LEFT": {
        "type": "DIRECTION",
        "canonical": "左转",
        "aliases": ["左转"],
        "fuzzy_patterns": [
            re.compile(r"^[左佐坐作]{1,2}[转专砖传赚]{1,2}$"),
        ],
    },
    "RIGHT": {
        "type": "DIRECTION",
        "canonical": "右转",
        "aliases": ["右转"],
        "fuzzy_patterns": [
            re.compile(r"^[右又佑幼]{1,2}[转专砖传赚]{1,2}$"),
        ],
    },
    "TURN_ON": {
        "type": "DISCRETE",
        "canonical": "启动",
        "aliases": ["启动", "开机"],
        "fuzzy_patterns": [],
    },
    "TURN_OFF": {
        "type": "DISCRETE",
        "canonical": "关机",
        "aliases": ["关机", "关闭", "休眠"],
        "fuzzy_patterns": [],
    },
    "ROLL": {
        "type": "DISCRETE",
        "canonical": "翻滚",
        "aliases": ["翻滚"],
        "fuzzy_patterns": [],
    },
    "DANCE": {
        "type": "DISCRETE",
        "canonical": "跳舞",
        "aliases": ["跳舞"],
        "fuzzy_patterns": [
            re.compile(r"^[跳条挑]{1,2}[舞五午]{1,2}$"),
        ],
    },
    "STOP": {
        "type": "DISCRETE",
        "canonical": "停止",
        "aliases": ["停止", "停"],
        "fuzzy_patterns": [
            re.compile(r"^[停庭婷廷](止)?$"),
        ],
    },
    "TWIST": {
        "type": "DISCRETE",
        "canonical": "扭",
        "aliases": ["扭", "扭动"],
        "fuzzy_patterns": [],
    },
    "FAST": {
        "type": "PARAMETER",
        "canonical": "加速",
        "aliases": ["加速", "快"],
        "fuzzy_patterns": [],
    },
    "SLOW": {
        "type": "PARAMETER",
        "canonical": "慢速",
        "aliases": ["慢速", "慢"],
        "fuzzy_patterns": [],
    },
}

ALL_ALIASES = []
for cmd, spec in COMMAND_SPECS.items():
    for alias in spec["aliases"]:
        ALL_ALIASES.append((cmd, alias))
ALL_ALIASES.sort(key=lambda x: len(x[1]), reverse=True)


def get_control_semantics_for_voice(command: str) -> str:
    """
    Day 1 先把语音命令统一标记为 DISCRETE。
    后续若加入真正的语音持续控制，再单独细化。
    """
    return CONTROL_SEMANTICS_DISCRETE


# ======================== 匹配逻辑 ========================
def exact_multi_scan(normalized_text: str):
    """
    只做 exact alias 扫描，用于“打招呼前进”这类连续命令。
    fuzzy 不参与多命令扫描，避免误触发。
    """
    hits = []
    occupied = []

    for cmd, alias in ALL_ALIASES:
        start = 0
        while True:
            pos = normalized_text.find(alias, start)
            if pos == -1:
                break
            end = pos + len(alias)

            overlap = False
            for s, e in occupied:
                if not (end <= s or pos >= e):
                    overlap = True
                    break

            if not overlap:
                hits.append((pos, end, cmd, alias))
                occupied.append((pos, end))

            start = pos + 1

    hits.sort(key=lambda x: (x[0], -(x[1] - x[0])))

    results = []
    for _, _, cmd, alias in hits:
        spec = COMMAND_SPECS[cmd]
        if not results or results[-1]["command"] != cmd:
            results.append({
                "command": cmd,
                "type": spec["type"],
                "canonical": spec["canonical"],
                "matched_text": alias,
                "mode": "exact_multi",
                "confidence": 0.98,
            })

    return results

def match_segment(seg: str):
    """
    严格匹配顺序：
    1. exact alias
    2. alias + 极短尾巴
    3. fuzzy pattern（仅短语段）
    """
    seg = canonicalize_segment(seg)
    if not seg:
        return None, seg

    # 1. exact
    for cmd, spec in COMMAND_SPECS.items():
        for alias in spec["aliases"]:
            if seg == alias:
                return {
                    "command": cmd,
                    "type": spec["type"],
                    "canonical": spec["canonical"],
                    "matched_text": alias,
                    "mode": "exact",
                    "confidence": 0.99,
                }, seg

    # 2. prefix + 1 字尾巴
    for cmd, spec in COMMAND_SPECS.items():
        for alias in spec["aliases"]:
            if len(alias) <= 1:
                continue
            if seg.startswith(alias):
                suffix = seg[len(alias):]
                if suffix and len(suffix) <= 1:
                    return {
                        "command": cmd,
                        "type": spec["type"],
                        "canonical": spec["canonical"],
                        "matched_text": alias,
                        "mode": "prefix",
                        "confidence": 0.93,
                    }, seg

    # 3. fuzzy：只允许短语段
    if len(seg) <= MAX_FUZZY_SEGMENT_LEN:
        for cmd, spec in COMMAND_SPECS.items():
            for pat in spec["fuzzy_patterns"]:
                if pat.fullmatch(seg):
                    return {
                        "command": cmd,
                        "type": spec["type"],
                        "canonical": spec["canonical"],
                        "matched_text": seg,
                        "mode": "fuzzy",
                        "confidence": 0.86,
                    }, seg

    return None, seg

def parse_commands(raw_text: str):
    normalized = normalize_text(raw_text)
    if not normalized:
        return [], normalized

    segments = [s for s in normalized.split(",") if s.strip()]
    if not segments:
        segments = [normalized]

    matches = []
    cleaned_segments = []

    for seg in segments:
        match, cleaned = match_segment(seg)
        cleaned_segments.append(cleaned)

        if match:
            if not matches or matches[-1]["command"] != match["command"]:
                matches.append(match)

    if not matches:
        matches = exact_multi_scan(normalized)

    if not ALLOW_MULTI_COMMANDS and matches:
        matches = [matches[0]]

    normalized_for_parse = ",".join([s for s in cleaned_segments if s])
    if not normalized_for_parse:
        normalized_for_parse = normalized

    return matches, normalized_for_parse


# ======================== termios 工具 ========================
_term_lock = threading.Lock()

def read_one_char():
    fd = sys.stdin.fileno()
    with _term_lock:
        old_settings = termios.tcgetattr(fd)
        try:
            tty.setraw(fd)
            ch = sys.stdin.read(1)
        finally:
            termios.tcsetattr(fd, termios.TCSAFLUSH, old_settings)
    return ch


# ======================== 输入设备选择 ========================
def choose_input_device(preferred_device_id=None):
    try:
        devices = sd.query_devices()
    except Exception as e:
        raise RuntimeError(f"查询 sounddevice 设备失败: {e}")

    valid_inputs = []
    for idx, dev in enumerate(devices):
        if dev.get("max_input_channels", 0) > 0:
            valid_inputs.append((idx, dev))

    if not valid_inputs:
        raise RuntimeError("未找到任何可用的输入音频设备。")

    if preferred_device_id is not None:
        try:
            dev = devices[preferred_device_id]
            if dev.get("max_input_channels", 0) > 0:
                return preferred_device_id, dev
        except Exception:
            pass

    try:
        default_input, _ = sd.default.device
        if default_input is not None and default_input >= 0:
            dev = devices[default_input]
            if dev.get("max_input_channels", 0) > 0:
                return default_input, dev
    except Exception:
        pass

    return valid_inputs[0][0], valid_inputs[0][1]


# ======================== 热词文件检查 ========================
def inspect_hotwords_file(hotwords_path: Path):
    info = {
        "exists": hotwords_path.is_file(),
        "utf8_ok": False,
        "lines": [],
        "warnings": [],
    }

    if not hotwords_path.is_file():
        info["warnings"].append(f"hotwords 文件不存在: {hotwords_path}")
        return info

    try:
        content = hotwords_path.read_text(encoding="utf-8").splitlines()
        info["utf8_ok"] = True
    except UnicodeDecodeError:
        info["warnings"].append(
            f"hotwords 文件不是 UTF-8 编码: {hotwords_path}，请改成 UTF-8 无 BOM。"
        )
        return info
    except Exception as e:
        info["warnings"].append(f"读取 hotwords 文件失败: {e}")
        return info

    for line in content:
        s = line.strip()
        if s and not s.startswith("#"):
            info["lines"].append(s)

    if not info["lines"]:
        info["warnings"].append("hotwords 文件为空，热词不会生效。")

    ascii_like = [x for x in info["lines"] if re.fullmatch(r"[A-Za-z\s'\-]+", x)]
    if ascii_like and len(ascii_like) >= max(1, len(info["lines"]) // 2):
        info["warnings"].append(
            "检测到 hotwords 多数为 ASCII / 拼音风格。"
            "若当前 tokens.txt 不是同一 tokenizer 体系，这些热词大概率无效。"
        )

    return info


# ======================== ROS 节点 ========================
class VoiceNode(Node):
    def __init__(self):
        super().__init__("voice_node")
        self.get_logger().info("语音节点启动中...")

        self.model_dir = resolve_model_dir()
        self.tokens_path = self.model_dir / "tokens.txt"
        self.encoder_path = self.model_dir / "encoder-epoch-99-avg-1.int8.onnx"
        self.decoder_path = self.model_dir / "decoder-epoch-99-avg-1.int8.onnx"
        self.joiner_path = self.model_dir / "joiner-epoch-99-avg-1.int8.onnx"
        self.hotwords_path = resolve_hotwords_file()

        self._log_paths()
        self._check_required_files()

        hotword_info = inspect_hotwords_file(self.hotwords_path)
        for w in hotword_info["warnings"]:
            self.get_logger().warn(w)

        self.hotwords_usable = (
            hotword_info["exists"] and
            hotword_info["utf8_ok"] and
            len(hotword_info["lines"]) > 0
        )

        if self.hotwords_usable:
            preview = " / ".join(hotword_info["lines"][:10])
            self.get_logger().info(f"hotwords 可用，预览: {preview}")
        else:
            self.get_logger().warn("hotwords 不可用，本次运行将禁用 hotwords。")

        if not _HAS_SCIPY:
            self.get_logger().warn(
                "scipy 未安装，降采样退回简单抽帧，识别率会下降。"
                "建议安装: pip3 install scipy --break-system-packages"
            )

        self.pub = self.create_publisher(String, "/voice_command", 10)
        self.raw_pub = None
        if PUBLISH_RAW_TEXT:
            self.raw_pub = self.create_publisher(String, RAW_TEXT_TOPIC, 10)

        self.recognizer = self._create_recognizer()
        self.stream = self.recognizer.create_stream()

        self._speaking_event = threading.Event()
        self._stream_lock = threading.Lock()
        self._publish_lock = threading.Lock()
        self._shutting_down = False
        self._allow_partial_print = False

        self.last_partial_text = ""
        self.last_partial_print_time = 0.0
        self.last_overflow_warn_time = 0.0
        self.last_queue_full_warn_time = 0.0

        self._process_thread = None

        self.audio_queue = queue.Queue(maxsize=AUDIO_QUEUE_MAXSIZE)
        self.audio_worker = threading.Thread(target=self._process_audio_queue, daemon=True)
        self.audio_worker.start()

        self.keyboard_thread = threading.Thread(target=self._listen_keyboard, daemon=True)
        self.keyboard_thread.start()

        self.device_id, dev_info = choose_input_device(MIC_DEVICE_ID)
        self.get_logger().info(
            f"打开麦克风: device_id={self.device_id}, "
            f"name={dev_info.get('name', 'unknown')}, "
            f"samplerate={MIC_SAMPLE_RATE}Hz"
        )

        self.audio_stream = sd.InputStream(
            samplerate=MIC_SAMPLE_RATE,
            device=self.device_id,
            channels=1,
            blocksize=AUDIO_BLOCKSIZE,
            latency=AUDIO_LATENCY,
            dtype="float32",
            callback=self._audio_callback,
        )
        self.audio_stream.start()

    # ------------------------ 启动检查 ------------------------
    def _log_paths(self):
        self.get_logger().info(f"cwd: {Path.cwd().resolve()}")
        self.get_logger().info(f"model_dir: {self.model_dir}")
        self.get_logger().info(f"tokens: {self.tokens_path}")
        self.get_logger().info(f"encoder: {self.encoder_path}")
        self.get_logger().info(f"decoder: {self.decoder_path}")
        self.get_logger().info(f"joiner: {self.joiner_path}")
        self.get_logger().info(f"hotwords: {self.hotwords_path}")

    def _check_required_files(self):
        required = [
            self.tokens_path,
            self.encoder_path,
            self.decoder_path,
            self.joiner_path,
        ]
        missing = [str(p) for p in required if not p.is_file()]
        if missing:
            raise FileNotFoundError(
                "以下模型文件缺失：\n" + "\n".join(missing)
            )

    # ------------------------ recognizer ------------------------
    def _create_recognizer(self):
        self.get_logger().info("加载 sherpa_onnx 模型...")

        kwargs = dict(
            tokens=str(self.tokens_path),
            encoder=str(self.encoder_path),
            decoder=str(self.decoder_path),
            joiner=str(self.joiner_path),
            num_threads=NUM_THREADS,
            provider=PROVIDER,
            sample_rate=RECOGNIZER_SAMPLE_RATE,
            feature_dim=FEATURE_DIM,
            decoding_method=DECODING_METHOD,
            max_active_paths=MAX_ACTIVE_PATHS,
            rule2_min_trailing_silence=RULE2_MIN_TRAILING,
            rule3_min_utterance_length=RULE3_MIN_UTTERANCE,
        )

        if self.hotwords_usable:
            kwargs["hotwords_file"] = str(self.hotwords_path)
            kwargs["hotwords_score"] = HOTWORDS_SCORE
            self.get_logger().info("热词文件有效，已启用 hotwords")
        else:
            self.get_logger().warn("热词文件不可用，已禁用 hotwords")

        recognizer = sherpa_onnx.OnlineRecognizer.from_transducer(**kwargs)
        self.get_logger().info("模型加载完成！")
        return recognizer

    # ------------------------ keyboard ------------------------
    def _clear_audio_queue(self):
        while not self.audio_queue.empty():
            try:
                self.audio_queue.get_nowait()
                self.audio_queue.task_done()
            except queue.Empty:
                break

    def _listen_keyboard(self):
        safe_print("\n" + "=" * 60)
        safe_print("💡 [操作说明] 请点击当前终端窗口：")
        safe_print("   👉 按【空格键】开始录音")
        safe_print("   👉 再按一次【空格键】结束录音并解析")
        safe_print("   👉 按【Ctrl+C】退出节点")
        safe_print("   👉 示例：打个招呼 / 前进 / 停 / 加速 / 打招呼，前进")
        safe_print("=" * 60 + "\n")

        while rclpy.ok() and not self._shutting_down:
            try:
                ch = read_one_char()

                if ch == "\x03":
                    raise KeyboardInterrupt

                if ch != " ":
                    continue

                if not self._speaking_event.is_set():
                    if self._process_thread is not None and self._process_thread.is_alive():
                        self._process_thread.join(timeout=3.0)

                    self._clear_audio_queue()

                    with self._stream_lock:
                        self.recognizer.reset(self.stream)

                    self.last_partial_text = ""
                    self.last_partial_print_time = 0.0

                    self._allow_partial_print = True
                    self._speaking_event.set()
                    safe_print("\n🎤 [录音中] 请说话...")
                else:
                    self._speaking_event.clear()
                    self._allow_partial_print = False
                    safe_print("\n📤 [已结束] 正在解析指令...")

                    t = threading.Thread(target=self._process_final_result, daemon=False)
                    self._process_thread = t
                    t.start()

            except KeyboardInterrupt:
                safe_print("\n🛑 检测到 Ctrl+C，正在退出...")
                os.kill(os.getpid(), signal.SIGINT)
                break
            except Exception as e:
                safe_print(f"\n⚠️ 键盘线程异常: {e}")
                traceback.print_exc()

    # ------------------------ 音频回调 / 后台线程 ------------------------
    def _audio_callback(self, indata, frames, time_info, status):
        if self._shutting_down or not self._speaking_event.is_set():
            return

        if status:
            now = time.monotonic()
            if now - self.last_overflow_warn_time > OVERFLOW_WARN_INTERVAL_SEC:
                self.get_logger().warn(f"audio status: {status}")
                self.last_overflow_warn_time = now

        try:
            self.audio_queue.put_nowait(indata[:, 0].copy())
        except queue.Full:
            now = time.monotonic()
            if now - self.last_queue_full_warn_time > QUEUE_FULL_WARN_INTERVAL_SEC:
                self.get_logger().warn("audio queue full，已丢弃一块音频数据")
                self.last_queue_full_warn_time = now

    def _process_audio_queue(self):
        while rclpy.ok() and not self._shutting_down:
            try:
                audio_chunk = self.audio_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            try:
                # 极轻门限：纯静音时不解码
                rms = float(np.sqrt(np.mean(audio_chunk ** 2)))
                if rms < PRE_RMS_GATE:
                    continue

                audio_16k = downsample_48k_to_16k(audio_chunk)

                with self._stream_lock:
                    self.stream.accept_waveform(RECOGNIZER_SAMPLE_RATE, audio_16k)

                    while self.recognizer.is_ready(self.stream):
                        self.recognizer.decode_stream(self.stream)

                    current_text = self.recognizer.get_result(self.stream).strip()

                if current_text and self._allow_partial_print:
                    display = normalize_text(current_text)
                    now = time.monotonic()
                    if (
                        display
                        and display != self.last_partial_text
                        and (now - self.last_partial_print_time) >= PARTIAL_PRINT_MIN_INTERVAL
                    ):
                        safe_write(f"\r\033[2K👂 [听到]: {display}")
                        self.last_partial_text = display
                        self.last_partial_print_time = now
            except Exception as e:
                self.get_logger().error(f"_process_audio_queue exception: {e}")
                traceback.print_exc()
            finally:
                self.audio_queue.task_done()

    # ------------------------ 发布 / 终解码 ------------------------
    def _publish_raw_text(self, text: str):
        if self.raw_pub is None:
            return
        try:
            msg = String()
            msg.data = text
            self.raw_pub.publish(msg)
        except Exception as e:
            self.get_logger().warn(f"发布原始文本失败: {e}")

    def _publish_matches_as_json(self, matches, raw_text, normalized_text, normalized_for_parse):
        with self._publish_lock:
            now_ts = time.time()
            for i, m in enumerate(matches):
                if self._shutting_down:
                    return

                payload = {
                    "modality": "VOICE",
                    "type": m["type"],
                    "value": m["command"],
                    "canonical": m["canonical"],
                    "matched_text": m["matched_text"],
                    "match_mode": m["mode"],
                    "confidence": round(float(m["confidence"]), 3),
                    "raw_text": raw_text,
                    "normalized_text": normalized_text,
                    "parsed_text": normalized_for_parse,
                    "timestamp": now_ts,
                    "control_semantics": get_control_semantics_for_voice(m["command"]),
                }

                msg = String()
                msg.data = json.dumps(payload, ensure_ascii=False)
                self.pub.publish(msg)

                safe_print(
                    f"🚀 指令已发布: {m['command']} "
                    f"({m['type']}, {m['mode']}, {payload.get('control_semantics', '-')}, conf={payload['confidence']})"
                )

                if i != len(matches) - 1 and COMMAND_PUBLISH_INTERVAL > 0:
                    time.sleep(COMMAND_PUBLISH_INTERVAL)

    def _wait_until_audio_queue_drained(self, timeout_sec=1.5):
        start = time.monotonic()
        while (time.monotonic() - start) < timeout_sec:
            if self.audio_queue.empty():
                break
            time.sleep(0.02)

    def _process_final_result(self):
        try:
            if self._shutting_down:
                return

            # 等后台线程先把停止前已入队的音频处理完
            self._wait_until_audio_queue_drained(timeout_sec=1.5)

            with self._stream_lock:
                tail = np.zeros(
                    int(RECOGNIZER_SAMPLE_RATE * FINAL_TAIL_SILENCE_SEC),
                    dtype=np.float32
                )
                self.stream.accept_waveform(RECOGNIZER_SAMPLE_RATE, tail)

                while self.recognizer.is_ready(self.stream):
                    self.recognizer.decode_stream(self.stream)

                raw_text = self.recognizer.get_result(self.stream).strip()

            if not raw_text:
                safe_print('\n⚠️ 没有识别到有效语音，请重试。')
                return

            cleaned = clean_text(raw_text)
            normalized = normalize_text(raw_text)
            matches, normalized_for_parse = parse_commands(raw_text)

            safe_print(f'\n📝 原始识别: "{cleaned}"')
            safe_print(f'✅ 归一化后: "{normalized}"')
            safe_print(f'🧹 匹配清洗后: "{normalized_for_parse}"')

            self._publish_raw_text(normalized_for_parse)

            if matches:
                safe_print(f"🎯 匹配命令: {[m['command'] for m in matches]}")
                self._publish_matches_as_json(
                    matches=matches,
                    raw_text=cleaned,
                    normalized_text=normalized,
                    normalized_for_parse=normalized_for_parse,
                )
            else:
                safe_print("❓ 未匹配到有效指令")

            safe_print("")
        except Exception as e:
            safe_print(f"\n❌ 解析指令时发生异常: {e}")
            traceback.print_exc()

    # ------------------------ 销毁 ------------------------
    def destroy_node(self):
        self._shutting_down = True
        self._speaking_event.clear()
        self._allow_partial_print = False

        try:
            if hasattr(self, "audio_stream"):
                self.audio_stream.stop()
                self.audio_stream.close()
        except Exception as e:
            self.get_logger().warn(f"关闭音频流失败: {e}")

        try:
            if self._process_thread is not None and self._process_thread.is_alive():
                self._process_thread.join(timeout=3.0)
        except Exception:
            pass

        try:
            if hasattr(self, "audio_worker") and self.audio_worker.is_alive():
                self.audio_worker.join(timeout=1.5)
        except Exception:
            pass

        super().destroy_node()


# ======================== main ========================
def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = VoiceNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        safe_print("\n🛑 节点被用户中断关闭")
    except Exception as e:
        safe_print(f"\n❌ 节点启动失败: {e}")
        traceback.print_exc()
    finally:
        if node is not None:
            node.destroy_node()
        try:
            rclpy.shutdown()
        except Exception:
            pass


if __name__ == "__main__":
    main()