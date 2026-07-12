#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
gesture_node.py
手势节点：通过 UDP 接收 ESP32 手套发来的数字指令或中文指令，
并按与 voice_node.py 一致的 JSON 控制接口发布到 /gesture_command。

"""

import json
import socket
import time
from typing import Optional, Tuple

import rclpy
from rclpy.node import Node
from std_msgs.msg import String

# ======================== UDP 配置 ========================
UDP_HOST = '0.0.0.0'
UDP_PORT = 9999
POLL_HZ = 20

# ======================== 发布配置 ========================
COMMAND_TOPIC = '/gesture_command'
RAW_TOPIC = '/gesture_text'
PUBLISH_RAW_TEXT = True
ALLOW_REPEAT_STOP = True

# ======================== 控制语义 ========================
CONTROL_SEMANTICS_CONTINUOUS = 'CONTINUOUS'
CONTROL_SEMANTICS_DISCRETE = 'DISCRETE'

# ======================== 指令定义 ========================
COMMAND_SPECS = {
    'HELLO': {
        'type': 'DISCRETE',
        'canonical': '打招呼',
        'aliases': ['打招呼', '你好'],
    },
    'FORWARD': {
        'type': 'DIRECTION',
        'canonical': '前进',
        'aliases': ['前进', '向前'],
    },
    'BACKWARD': {
        'type': 'DIRECTION',
        'canonical': '后退',
        'aliases': ['后退', '向后'],
    },
    'LEFT_MOVE': {
        'type': 'DIRECTION',
        'canonical': '左移',
        'aliases': ['左移'],
    },
    'RIGHT_MOVE': {
        'type': 'DIRECTION',
        'canonical': '右移',
        'aliases': ['右移'],
    },
    'LEFT': {
        'type': 'DIRECTION',
        'canonical': '左转',
        'aliases': ['左转'],
    },
    'RIGHT': {
        'type': 'DIRECTION',
        'canonical': '右转',
        'aliases': ['右转'],
    },
    'TURN_ON': {
        'type': 'DISCRETE',
        'canonical': '启动',
        'aliases': ['启动', '开机'],
    },
    'TURN_OFF': {
        'type': 'DISCRETE',
        'canonical': '关机',
        'aliases': ['关机', '关闭', '休眠'],
    },
    'ROLL': {
        'type': 'DISCRETE',
        'canonical': '翻滚',
        'aliases': ['翻滚'],
    },
    'DANCE': {
        'type': 'DISCRETE',
        'canonical': '跳舞',
        'aliases': ['跳舞'],
    },
    'STOP': {
        'type': 'DISCRETE',
        'canonical': '停止',
        'aliases': ['停止', '停'],
    },
    'TWIST': {
        'type': 'DISCRETE',
        'canonical': '扭动',
        'aliases': ['扭', '扭动'],
    },
}

# ESP32 数字指令映射（来自 esp32_control.cpp robotDogControl()）
NUM_CMD_MAP = {
    '1': 'STOP',
    '3': 'TURN_OFF',
    '4': 'TURN_ON',
    '5': 'STOP',
    '6': 'FORWARD',
    '7': 'BACKWARD',
    '8': 'LEFT_MOVE',
    '9': 'RIGHT_MOVE',
    '10': 'LEFT',
    '11': 'RIGHT',
    '12': 'ROLL',
    '13': 'DANCE',
    '14': 'TWIST',
    '15': 'HELLO',
}

# 中文数据源兼容
TEXT_CMD_MAP = {
    '前进': 'FORWARD',
    '向前': 'FORWARD',
    '后退': 'BACKWARD',
    '向后': 'BACKWARD',
    '左转': 'LEFT',
    '右转': 'RIGHT',
    '左移': 'LEFT_MOVE',
    '右移': 'RIGHT_MOVE',
    '停止': 'STOP',
    '停': 'STOP',
    '启动': 'TURN_ON',
    '开机': 'TURN_ON',
    '关机': 'TURN_OFF',
    '关闭': 'TURN_OFF',
    '休眠': 'TURN_OFF',
    '跳舞': 'DANCE',
    '翻滚': 'ROLL',
    '你好': 'HELLO',
    '打招呼': 'HELLO',
    '扭': 'TWIST',
    '扭动': 'TWIST',
}


DIRECTION_COMMANDS = {
    'FORWARD', 'BACKWARD', 'LEFT_MOVE', 'RIGHT_MOVE', 'LEFT', 'RIGHT'
}


def get_control_semantics(command: str) -> str:
    if command in DIRECTION_COMMANDS:
        return CONTROL_SEMANTICS_CONTINUOUS
    return CONTROL_SEMANTICS_DISCRETE


def normalize_text(text: str) -> str:
    text = text.strip()
    text = text.replace('，', ',').replace('。', ',').replace('！', ',').replace('？', ',')
    text = text.replace('；', ',').replace('、', ',').replace('!', ',').replace('?', ',')
    text = ''.join(text.split())
    while ',,' in text:
        text = text.replace(',,', ',')
    return text.strip(',')


def build_payload(
    *,
    command: str,
    raw_text: str,
    normalized_text: str,
    matched_text: str,
    match_mode: str,
    confidence: float,
    source_addr: Optional[str] = None,
) -> dict:
    spec = COMMAND_SPECS[command]
    payload = {
        'modality': 'GESTURE',
        'type': spec['type'],
        'value': command,
        'canonical': spec['canonical'],
        'matched_text': matched_text,
        'match_mode': match_mode,
        'confidence': round(float(confidence), 3),
        'raw_text': raw_text,
        'normalized_text': normalized_text,
        'parsed_text': normalized_text,
        'timestamp': time.time(),
        'control_semantics': get_control_semantics(command),
    }
    if source_addr:
        payload['source_addr'] = source_addr
    return payload


def parse_incoming_command(raw: str) -> Tuple[Optional[dict], Optional[str]]:
    """
    返回: (payload, normalized_text)
    - payload 为统一后的 JSON 对象
    - normalized_text 供调试通道发布
    """
    raw = raw.strip()
    if not raw:
        return None, None

    # 1. 若上游已经直接发来 JSON，则尽量兼容透传
    if raw.startswith('{') and raw.endswith('}'):
        try:
            data = json.loads(raw)
            value = str(data.get('value', '')).strip().upper()
            if value in COMMAND_SPECS:
                spec = COMMAND_SPECS[value]
                payload = {
                    'modality': data.get('modality', 'GESTURE'),
                    'type': data.get('type', spec['type']),
                    'value': value,
                    'canonical': data.get('canonical', spec['canonical']),
                    'matched_text': data.get('matched_text', raw),
                    'match_mode': data.get('match_mode', 'already_json'),
                    'confidence': round(float(data.get('confidence', 0.99)), 3),
                    'raw_text': data.get('raw_text', raw),
                    'normalized_text': data.get('normalized_text', spec['canonical']),
                    'parsed_text': data.get('parsed_text', spec['canonical']),
                    'timestamp': float(data.get('timestamp', time.time())),
                    'control_semantics': data.get('control_semantics', get_control_semantics(value)),
                }
                if 'source_addr' in data:
                    payload['source_addr'] = data['source_addr']
                return payload, str(payload['parsed_text'])
        except Exception:
            pass

    # 2. 数字映射
    if raw in NUM_CMD_MAP:
        cmd = NUM_CMD_MAP[raw]
        payload = build_payload(
            command=cmd,
            raw_text=raw,
            normalized_text=COMMAND_SPECS[cmd]['canonical'],
            matched_text=raw,
            match_mode='map_numeric',
            confidence=0.99,
        )
        return payload, payload['parsed_text']

    # 3. 中文 / 文本映射
    normalized = normalize_text(raw)
    if normalized in TEXT_CMD_MAP:
        cmd = TEXT_CMD_MAP[normalized]
        payload = build_payload(
            command=cmd,
            raw_text=raw,
            normalized_text=normalized,
            matched_text=normalized,
            match_mode='map_text',
            confidence=0.96,
        )
        return payload, payload['parsed_text']

    return None, normalized


class GestureNode(Node):
    def __init__(self):
        super().__init__('gesture_node')
        self.get_logger().info('手势节点启动中...[DAY1_NEW_BUILD_CHECK]')

        self.pub = self.create_publisher(String, COMMAND_TOPIC, 10)
        self.raw_pub = None
        if PUBLISH_RAW_TEXT:
            self.raw_pub = self.create_publisher(String, RAW_TOPIC, 10)

        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setblocking(False)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.bind((UDP_HOST, UDP_PORT))
        self.get_logger().info(f'UDP 监听: {UDP_HOST}:{UDP_PORT}')
        self.get_logger().info(f'统一控制接口话题: {COMMAND_TOPIC} ...')
        self._last_command = None
        self.timer = self.create_timer(1.0 / POLL_HZ, self._poll_udp)

        self.get_logger().info('手势节点就绪，等待手套数据...')

    def _publish_raw_text(self, text: Optional[str]):
        if self.raw_pub is None or not text:
            return
        msg = String()
        msg.data = text
        self.raw_pub.publish(msg)

    def _publish_payload(self, payload: dict):
        msg = String()
        msg.data = json.dumps(payload, ensure_ascii=False)
        self.pub.publish(msg)
        self.get_logger().info(
            f"🧤 手势已发布: {payload['value']} "
            f"({payload['type']}, {payload['match_mode']}, {payload.get('control_semantics', '-')}, conf={payload['confidence']})"
        )

    def _poll_udp(self):
        try:
            data, addr = self.sock.recvfrom(1024)
        except BlockingIOError:
            return
        except Exception as e:
            self.get_logger().error(f'UDP 异常: {e}')
            return

        raw = data.decode('utf-8', errors='ignore').strip()
        if not raw:
            return

        payload, normalized = parse_incoming_command(raw)
        self._publish_raw_text(normalized)

        if payload is None:
            # 传感器连续原始数据（如 "0.12 -0.34 ..."）直接忽略
            self.get_logger().debug(f'忽略非指令数据: "{raw}" 来自 {addr}')
            return

        payload['source_addr'] = f'{addr[0]}:{addr[1]}'
        cmd = payload['value']

        # Day 1 保持原有防抖策略不变：相同指令不重复发布；STOP 可重复发布。
        if cmd == self._last_command and not (ALLOW_REPEAT_STOP and cmd == 'STOP'):
            return

        self._publish_payload(payload)
        self._last_command = cmd

    def destroy_node(self):
        try:
            self.sock.close()
            self.get_logger().info('UDP Socket 已关闭')
        except Exception:
            pass
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = GestureNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('手势节点被用户中断')
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
