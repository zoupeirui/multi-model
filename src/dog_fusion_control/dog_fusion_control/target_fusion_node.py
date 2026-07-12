#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
eye_vision_fusion_node.py
=========================
眼动 + 视觉 空间融合节点（第 9 号节点）。

【职责】
  订阅 /eye/direction（眼动方向）和 /vision/objects（YOLO 检测 JSON），
  在滑动窗口内做「凝视判定 + 空间匹配」，锁定控制目标（DOG / CAR），
  发布到 /target_locked（规范话题）并桥接到 /eye_target（驱动现有 fusion_node）。

【输入话题】
  /eye/direction   std_msgs/String   裸字符串: "LEFT" | "CENTER" | "RIGHT" | "INVALID"
                                     （来自 eye_direction_node，~30Hz）
  /vision/objects  std_msgs/String   JSON:
      {
        "modality": "VISION", "type": "OBJECT_DETECTION",
        "image_width": 1280, "image_height": 720,
        "objects": [
          {"label": "robot_dog"|"robot_car", "confidence": 0.92,
           "bbox": [x1,y1,x2,y2], "x_center": 481, "y_center": 359}
        ]
      }
      （来自 vision_bridge_node，转发板 B 的 UDP）

【输出话题】
  /target_locked   std_msgs/String   JSON，value = DOG | CAR （符合 md 文档规范）
  /eye_target      std_msgs/String   JSON，value = DOG | VEHICLE （驱动 fusion_node_polished.py）
  /target_locked/debug  std_msgs/String  JSON 诊断信息（凝视比例、视觉持续度，便于调参）

  输出 JSON 示例（/target_locked）：
    {
      "modality": "FUSION", "type": "TARGET", "value": "DOG",
      "control_semantics": "DISCRETE", "confidence": 0.85,
      "decision_reason": "eye=LEFT 持续 1.6s(0.87) + 左侧检测 robot_dog(命中0.80, 均conf0.91)",
      "timestamp": 1747300000.123
    }

【核心逻辑】
  1. 维护过去 dwell_sec(默认1.5s) 的眼动方向滑动窗口和视觉检测滑动窗口。
  2. 眼动稳定判定：窗口内 LEFT 或 RIGHT 帧占比 >= stable_ratio(默认0.8)。
  3. 空间匹配：在选定一侧（x_center 相对 image_width 的左/右半区），
     某个 label 在 >= vision_persist_ratio(默认0.6) 比例的视觉帧中被检测到。
  4. label → 目标：robot_dog→DOG，robot_car→CAR。
  5. 切换冷却 switch_cooldown_sec(默认2.0s) 防止反复切换。
  6. 锁定变化时大声播报；并按 heartbeat_sec 静默重播当前目标（下游有去抖）。

【设计上的兼容性说明】
  - md 文档规范要求话题 /target_locked、值 DOG/CAR；
    但现有 fusion_node_polished.py 实际订阅 /eye_target、值 DOG/VEHICLE。
    本节点同时发布两者，CAR 在桥接到 /eye_target 时映射为 VEHICLE。
  - fusion_node_polished.py 默认 eye_gating_enabled=False，会忽略 /eye_target。
    若要让目标切换真正生效，请把该字段置 True（详见对话说明）。

【运行】
  # 直接运行（项目风格，容器内）
  source /opt/ros/foxy/install/setup.bash
  python3 eye_vision_fusion_node.py
  # 带参数
  python3 eye_vision_fusion_node.py --ros-args -p dwell_sec:=1.5 -p stable_ratio:=0.8

  Python 3.6 兼容（容器内 Foxy）；不依赖 dataclasses / 第三方库。
"""

import json
import time
from collections import deque, Counter
from typing import Optional, Dict, Any, List, Tuple

import rclpy
from rclpy.node import Node
from std_msgs.msg import String


# ======================== 话题默认值 ========================
DEFAULT_EYE_TOPIC          = '/eye/direction'
DEFAULT_VISION_TOPIC       = '/vision/objects'
DEFAULT_TARGET_TOPIC       = '/target_locked'      # 规范话题
DEFAULT_EYE_TARGET_TOPIC   = '/eye_target'         # 现有 fusion_node 实际订阅
DEFAULT_DEBUG_TOPIC        = '/target_locked/debug'

# ======================== label → 目标 映射 ========================
# YOLO 标准化标签 → 规范目标值（/target_locked 用）
LABEL_TO_TARGET = {
    'robot_dog': 'DOG',
    'robot_car': 'CAR',
}
# 规范目标值 → fusion_node_polished.py 接受的值（/eye_target 用）
TARGET_TO_EYE_TARGET = {
    'DOG': 'DOG',
    'CAR': 'VEHICLE',
}

# 眼动有效方向
EYE_VALID_SIDES = {'LEFT', 'RIGHT'}


class EyeVisionFusionNode(Node):
    """眼动 + 视觉 空间融合 → 目标锁定。"""

    def __init__(self):
        super().__init__('eye_vision_fusion_node')

        # ---------------- 参数声明 ----------------
        self.declare_parameter('eye_topic',         DEFAULT_EYE_TOPIC)
        self.declare_parameter('vision_topic',      DEFAULT_VISION_TOPIC)
        self.declare_parameter('target_topic',      DEFAULT_TARGET_TOPIC)
        self.declare_parameter('eye_target_topic',  DEFAULT_EYE_TARGET_TOPIC)
        self.declare_parameter('debug_topic',       DEFAULT_DEBUG_TOPIC)

        self.declare_parameter('dwell_sec',            1.5)    # 凝视/滑动窗口时长
        self.declare_parameter('stable_ratio',         0.8)    # 眼动方向稳定占比阈值
        self.declare_parameter('vision_persist_ratio', 0.6)    # 目标侧视觉持续命中占比阈值
        self.declare_parameter('center_deadband_frac', 0.05)   # 中心死区（占宽度比例，左右各半）
        self.declare_parameter('switch_cooldown_sec',  2.0)    # 目标切换冷却
        self.declare_parameter('min_eye_samples',      8)      # 决策所需最少眼动样本
        self.declare_parameter('min_vision_frames',    3)      # 决策所需最少视觉帧
        self.declare_parameter('decision_rate_hz',     5.0)    # 决策频率
        self.declare_parameter('heartbeat_sec',        1.0)    # 锁定后静默重播间隔(0=禁用)
        self.declare_parameter('invert_horizontal',    False)  # 摄像头左右镜像时翻转语义
        self.declare_parameter('enable_eye_target_bridge', True)  # 是否桥接到 /eye_target
        self.declare_parameter('publish_debug',        True)

        gp = lambda n: self.get_parameter(n).value
        self.eye_topic         = gp('eye_topic')
        self.vision_topic      = gp('vision_topic')
        self.target_topic      = gp('target_topic')
        self.eye_target_topic  = gp('eye_target_topic')
        self.debug_topic       = gp('debug_topic')

        self.dwell_sec            = float(gp('dwell_sec'))
        self.stable_ratio         = float(gp('stable_ratio'))
        self.vision_persist_ratio = float(gp('vision_persist_ratio'))
        self.center_deadband_frac = float(gp('center_deadband_frac'))
        self.switch_cooldown_sec  = float(gp('switch_cooldown_sec'))
        self.min_eye_samples      = int(gp('min_eye_samples'))
        self.min_vision_frames    = int(gp('min_vision_frames'))
        self.decision_rate_hz     = float(gp('decision_rate_hz'))
        self.heartbeat_sec        = float(gp('heartbeat_sec'))
        self.invert_horizontal    = bool(gp('invert_horizontal'))
        self.enable_eye_bridge    = bool(gp('enable_eye_target_bridge'))
        self.publish_debug        = bool(gp('publish_debug'))

        # ---------------- 滑动窗口 ----------------
        # 眼动: (recv_time, label)
        self._eye_win: deque = deque()
        # 视觉: (recv_time, {'LEFT': [(label, conf), ...], 'RIGHT': [...]})
        self._vision_win: deque = deque()

        # ---------------- 状态 ----------------
        self._locked_target: Optional[str] = None     # 当前锁定 DOG / CAR
        self._last_switch_ts: float = 0.0
        self._last_heartbeat_ts: float = 0.0

        # ---------------- 发布者 ----------------
        self.pub_target = self.create_publisher(String, self.target_topic, 10)
        self.pub_eye_target = (
            self.create_publisher(String, self.eye_target_topic, 10)
            if self.enable_eye_bridge else None
        )
        self.pub_debug = (
            self.create_publisher(String, self.debug_topic, 10)
            if self.publish_debug else None
        )

        # ---------------- 订阅者 ----------------
        self.create_subscription(String, self.eye_topic,    self._on_eye,    20)
        self.create_subscription(String, self.vision_topic, self._on_vision, 20)

        # ---------------- 决策定时器 ----------------
        self.create_timer(1.0 / max(self.decision_rate_hz, 1e-3), self._on_decide)

        self.get_logger().info('=' * 64)
        self.get_logger().info('👁️🤖 眼动-视觉融合节点已启动')
        self.get_logger().info('   订阅: %s (眼动)  +  %s (视觉)' % (self.eye_topic, self.vision_topic))
        self.get_logger().info('   发布: %s  +  %s%s'
                               % (self.target_topic,
                                  self.eye_target_topic if self.enable_eye_bridge else '(无桥接)',
                                  '' if self.enable_eye_bridge else ''))
        self.get_logger().info('   窗口=%.1fs  眼动稳定阈值=%.2f  视觉持续阈值=%.2f'
                               % (self.dwell_sec, self.stable_ratio, self.vision_persist_ratio))
        self.get_logger().info('   切换冷却=%.1fs  中心死区=%.2f  水平翻转=%s'
                               % (self.switch_cooldown_sec, self.center_deadband_frac, self.invert_horizontal))
        self.get_logger().info('=' * 64)

    # ================================================================
    #  眼动回调 —— 裸字符串
    # ================================================================
    def _on_eye(self, msg: String):
        label = (msg.data or '').strip().upper()
        if not label:
            return
        # INVALID / CENTER 也入窗（作为「非稳定方向」证据），但只保留我们关心的标签
        if label not in ('LEFT', 'RIGHT', 'CENTER', 'INVALID'):
            self.get_logger().debug('忽略未知眼动标签: %s' % label)
            return
        self._eye_win.append((time.time(), label))

    # ================================================================
    #  视觉回调 —— JSON
    # ================================================================
    def _on_vision(self, msg: String):
        raw = (msg.data or '').strip()
        if not raw or not raw.startswith('{'):
            return
        try:
            data = json.loads(raw)
        except json.JSONDecodeError as e:
            self.get_logger().warn('视觉 JSON 解析失败: %s' % e)
            return

        img_w = float(data.get('image_width', 0) or 0)
        objects = data.get('objects', []) or []
        if img_w <= 0:
            # 没有宽度无法做左右判定，跳过这一帧
            return

        mid = img_w / 2.0
        dead = img_w * self.center_deadband_frac  # 中心死区半宽
        sides = {'LEFT': [], 'RIGHT': []}

        for obj in objects:
            label = str(obj.get('label', '')).strip()
            if label not in LABEL_TO_TARGET:
                continue
            xc = obj.get('x_center', None)
            if xc is None:
                bbox = obj.get('bbox', None)
                if bbox and len(bbox) == 4:
                    xc = (float(bbox[0]) + float(bbox[2])) / 2.0
                else:
                    continue
            xc = float(xc)
            conf = float(obj.get('confidence', 0.0))

            # 左右判定（含中心死区）
            if xc < mid - dead:
                side = 'LEFT'
            elif xc > mid + dead:
                side = 'RIGHT'
            else:
                continue  # 落在中心死区，方向不明确，忽略

            if self.invert_horizontal:
                side = 'RIGHT' if side == 'LEFT' else 'LEFT'

            sides[side].append((label, conf))

        self._vision_win.append((time.time(), sides))

    # ================================================================
    #  窗口裁剪
    # ================================================================
    def _prune(self, now: float):
        cutoff = now - self.dwell_sec
        while self._eye_win and self._eye_win[0][0] < cutoff:
            self._eye_win.popleft()
        while self._vision_win and self._vision_win[0][0] < cutoff:
            self._vision_win.popleft()

    # ================================================================
    #  眼动稳定方向
    # ================================================================
    def _eye_stable_side(self) -> Tuple[Optional[str], float, int]:
        """返回 (稳定方向 LEFT/RIGHT 或 None, 该方向占比, 样本数)。"""
        n = len(self._eye_win)
        if n == 0:
            return None, 0.0, 0
        counts = Counter(lab for _, lab in self._eye_win)
        left_r = counts.get('LEFT', 0) / float(n)
        right_r = counts.get('RIGHT', 0) / float(n)
        if left_r >= self.stable_ratio and left_r >= right_r:
            return 'LEFT', left_r, n
        if right_r >= self.stable_ratio:
            return 'RIGHT', right_r, n
        # 返回较大者比例用于调试
        return None, max(left_r, right_r), n

    # ================================================================
    #  目标侧视觉持续命中
    # ================================================================
    def _vision_target_on_side(self, side: str) -> Tuple[Optional[str], float, float, int]:
        """
        在指定一侧统计哪个 label 持续出现。
        返回 (最佳label 或 None, 命中帧占比, 该label均置信度, 视觉帧数)。
        """
        vf = len(self._vision_win)
        if vf == 0:
            return None, 0.0, 0.0, 0

        presence = Counter()           # label -> 出现的帧数
        conf_sum: Dict[str, float] = {}
        conf_cnt: Dict[str, int] = {}

        for _, sides in self._vision_win:
            seen_in_frame = set()
            for (label, conf) in sides.get(side, []):
                if label not in seen_in_frame:
                    presence[label] += 1
                    seen_in_frame.add(label)
                conf_sum[label] = conf_sum.get(label, 0.0) + conf
                conf_cnt[label] = conf_cnt.get(label, 0) + 1

        if not presence:
            return None, 0.0, 0.0, vf

        # 选命中帧最多的 label；并列时取均置信度更高者
        def _score(item):
            label, cnt = item
            avg_conf = conf_sum[label] / max(conf_cnt[label], 1)
            return (cnt, avg_conf)

        best_label, best_cnt = max(presence.items(), key=lambda it: _score(it))
        hit_ratio = best_cnt / float(vf)
        avg_conf = conf_sum[best_label] / max(conf_cnt[best_label], 1)
        return best_label, hit_ratio, avg_conf, vf

    # ================================================================
    #  决策主循环
    # ================================================================
    def _on_decide(self):
        now = time.time()
        self._prune(now)

        n_eye = len(self._eye_win)
        n_vis = len(self._vision_win)

        # 数据不足
        if n_eye < self.min_eye_samples or n_vis < self.min_vision_frames:
            self._publish_debug('collecting',
                                eye_samples=n_eye, vision_frames=n_vis)
            return

        side, eye_ratio, _ = self._eye_stable_side()
        if side is None:
            self._publish_debug('eye_unstable',
                                eye_samples=n_eye, vision_frames=n_vis,
                                eye_ratio=round(eye_ratio, 3))
            self._maybe_heartbeat(now)
            return

        best_label, hit_ratio, avg_conf, _ = self._vision_target_on_side(side)
        if best_label is None or hit_ratio < self.vision_persist_ratio:
            self._publish_debug('no_persistent_object_on_side',
                                side=side, eye_ratio=round(eye_ratio, 3),
                                vision_frames=n_vis,
                                best_label=best_label,
                                hit_ratio=round(hit_ratio, 3))
            self._maybe_heartbeat(now)
            return

        target = LABEL_TO_TARGET.get(best_label)
        if target is None:
            return

        # 综合置信度：眼动稳定度 + 视觉命中率 + 视觉均置信度
        confidence = round(0.4 * eye_ratio + 0.3 * hit_ratio + 0.3 * avg_conf, 3)
        reason = ('eye=%s 持续%.1fs(%.2f) + %s侧检测 %s(命中%.2f, 均conf%.2f)'
                  % (side, self.dwell_sec, eye_ratio, side, best_label, hit_ratio, avg_conf))

        # ---- 锁定 / 切换判定 ----
        if target == self._locked_target:
            # 目标未变：心跳重播
            self._publish_debug('locked_stable', side=side, target=target,
                                confidence=confidence, eye_ratio=round(eye_ratio, 3),
                                hit_ratio=round(hit_ratio, 3))
            self._maybe_heartbeat(now, confidence=confidence, reason=reason)
            return

        # 目标发生变化 —— 检查冷却
        if (now - self._last_switch_ts) < self.switch_cooldown_sec and self._locked_target is not None:
            remain = self.switch_cooldown_sec - (now - self._last_switch_ts)
            self._publish_debug('switch_cooldown', from_target=self._locked_target,
                                to_target=target, cooldown_remain=round(remain, 2))
            return

        old = self._locked_target
        self._locked_target = target
        self._last_switch_ts = now
        self._last_heartbeat_ts = now

        self.get_logger().info('🔒 [锁定目标] %s → %s | conf=%.2f | %s'
                               % (old if old else 'NONE', target, confidence, reason))
        self._emit_target(target, confidence, reason)
        self._publish_debug('locked_change', from_target=old, to_target=target,
                            confidence=confidence, eye_ratio=round(eye_ratio, 3),
                            hit_ratio=round(hit_ratio, 3))

    # ================================================================
    #  心跳重播（下游有去抖，安全）
    # ================================================================
    def _maybe_heartbeat(self, now: float, confidence: float = 0.8, reason: str = 'heartbeat'):
        if self.heartbeat_sec <= 0.0 or self._locked_target is None:
            return
        if (now - self._last_heartbeat_ts) >= self.heartbeat_sec:
            self._last_heartbeat_ts = now
            self._emit_target(self._locked_target, confidence,
                              reason if reason != 'heartbeat' else
                              ('心跳重播_当前锁定_%s' % self._locked_target),
                              quiet=True)

    # ================================================================
    #  发布目标（同时写 /target_locked 与 /eye_target）
    # ================================================================
    def _emit_target(self, target: str, confidence: float, reason: str, quiet: bool = False):
        ts = time.time()

        # 1) 规范话题 /target_locked：value = DOG / CAR
        payload = {
            'modality': 'FUSION',
            'type': 'TARGET',
            'value': target,
            'control_semantics': 'DISCRETE',
            'confidence': confidence,
            'decision_reason': reason,
            'timestamp': ts,
        }
        msg = String()
        msg.data = json.dumps(payload, ensure_ascii=False)
        self.pub_target.publish(msg)

        # 2) 桥接话题 /eye_target：value = DOG / VEHICLE（驱动 fusion_node_polished.py）
        if self.pub_eye_target is not None:
            eye_val = TARGET_TO_EYE_TARGET.get(target, target)
            bridge = dict(payload)
            bridge['modality'] = 'EYE'
            bridge['value'] = eye_val
            bmsg = String()
            bmsg.data = json.dumps(bridge, ensure_ascii=False)
            self.pub_eye_target.publish(bmsg)

        if not quiet:
            self.get_logger().info('📤 [发布] target=%s (eye_target=%s) conf=%.2f'
                                   % (target, TARGET_TO_EYE_TARGET.get(target, target), confidence))

    # ================================================================
    #  调试发布
    # ================================================================
    def _publish_debug(self, state: str, **kw):
        if self.pub_debug is None:
            return
        payload = {'state': state, 'locked': self._locked_target, 'timestamp': time.time()}
        payload.update(kw)
        m = String()
        m.data = json.dumps(payload, ensure_ascii=False)
        self.pub_debug.publish(m)


def main(args=None):
    rclpy.init(args=args)
    node = EyeVisionFusionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('\n🛑 眼动-视觉融合节点被用户中断')
    except Exception as e:
        node.get_logger().error('节点崩溃: %s' % e)
        raise
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
