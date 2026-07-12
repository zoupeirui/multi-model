import json
import time
from typing import Optional, Dict, Any

import rclpy
from rclpy.node import Node
from std_msgs.msg import String

# ======================== 话题配置 ========================
VOICE_TOPIC      = '/voice_command'
GESTURE_TOPIC    = '/gesture_command'
EYE_TOPIC        = '/eye_target'

OUTPUT_TOPIC     = '/dog_command'
CAR_OUTPUT_TOPIC = '/car_command'
FEEDBACK_TOPIC   = '/fusion_feedback'

# ======================== 时间参数 ========================
DIRECTION_TIMEOUT_SEC   = 1.0    # 连续方向超时（无新输入自动 STANDBY）
VOICE_DEBOUNCE_SEC      = 1.0    # 语音去抖（相同指令重复间隔）
GESTURE_DEBOUNCE_SEC    = 0.3    # 手势去抖
EYE_TARGET_DEBOUNCE_SEC = 0.8    # 眼动目标切换去抖
OUTPUT_HZ               = 10     # 周期输出频率

# ======================== 模态常量 ========================
MODALITY_VOICE   = 'VOICE'
MODALITY_GESTURE = 'GESTURE'
MODALITY_EYE     = 'EYE'
MODALITY_FUSION  = 'FUSION'

# ======================== 指令分类 ========================
DIRECTION_COMMANDS = {'FORWARD', 'BACKWARD', 'LEFT', 'RIGHT', 'LEFT_MOVE', 'RIGHT_MOVE'}
DISCRETE_COMMANDS  = {'HELLO', 'DANCE', 'ROLL', 'TWIST', 'TURN_ON', 'TURN_OFF', 'ACCELERATE', 'DECELERATE'}
STOP_COMMANDS      = {'STOP'}

# ======================== 目标门控配置 ========================
TARGET_VALUES          = {'DOG', 'VEHICLE', 'NONE'}
DOG_ACTIVE_TARGETS     = {'DOG'}
VEHICLE_ACTIVE_TARGETS = {'VEHICLE'}

# ★ FIX Bug1: 默认目标改为 DOG，去掉重复赋值
DEFAULT_TARGET = 'DOG'

# ======================== 优先级（冲突仲裁用）========================
MODALITY_PRIORITY = {
    MODALITY_VOICE:   2,
    MODALITY_GESTURE: 1,
    MODALITY_FUSION:  3,
}


class FusionNode(Node):
 
    def __init__(self):
        super().__init__('fusion_node')

        # ---- 发布者 ----
        self.command_pub     = self.create_publisher(String, OUTPUT_TOPIC,     10)
        self.car_command_pub = self.create_publisher(String, CAR_OUTPUT_TOPIC, 10)
        self.feedback_pub    = self.create_publisher(String, FEEDBACK_TOPIC,   10)

        # ---- 订阅者 ----
        self.create_subscription(String, VOICE_TOPIC,   self._on_voice,      10)
        self.create_subscription(String, GESTURE_TOPIC, self._on_gesture,    10)
        self.create_subscription(String, EYE_TOPIC,     self._on_eye_target, 10)

        # ---- 状态机变量 ----
        self.active_direction    : Optional[str] = None
        self.active_direction_ts : float         = 0.0
        self.active_modality     : Optional[str] = None
        self.active_confidence   : float         = 0.0

        # ★ FIX Bug1: 只赋值一次，默认 DOG
        self.current_target    : str   = DEFAULT_TARGET
        self.current_target_ts : float = time.time()

        # ---- 去抖变量 ----
        self.last_voice_val   : str   = ''
        self.last_voice_ts    : float = 0.0
        self.last_gesture_val : str   = ''
        self.last_gesture_ts  : float = 0.0
        self.last_eye_target  : str   = ''
        # ★ FIX Bug3: 初始化为 0，使第一次眼动消息必定通过去抖
        self.last_eye_ts      : float = 0.0

        # ---- 待发离散动作 ----
        self.pending_discrete : Optional[Dict[str, Any]] = None

        # ---- 周期定时器 (10Hz) ----
        self.create_timer(1.0 / OUTPUT_HZ, self._on_timer)

        self.get_logger().info('=' * 60)
        self.get_logger().info('🧠 融合节点（已修复版）已启动')
        self.get_logger().info(f'   方向超时: {DIRECTION_TIMEOUT_SEC}s')
        self.get_logger().info(f'   语音去抖: {VOICE_DEBOUNCE_SEC}s')
        self.get_logger().info(f'   眼动目标去抖: {EYE_TARGET_DEBOUNCE_SEC}s')
        self.get_logger().info(f'   当前默认目标: {self.current_target}')
        self.get_logger().info('=' * 60)

    # ================================================================
    #  输入回调
    # ================================================================
    def _on_voice(self, msg: String):
        payload = self._parse_input(msg.data, MODALITY_VOICE)
        if payload is None:
            return

        val = payload['value']
        now = payload['timestamp']

        if val == self.last_voice_val and (now - self.last_voice_ts) < VOICE_DEBOUNCE_SEC:
            self.get_logger().debug(f'[去抖] 语音重复指令忽略: {val}')
            return

        self.last_voice_val = val
        self.last_voice_ts  = now
        self._handle_input(payload)

    def _on_gesture(self, msg: String):
        payload = self._parse_input(msg.data, MODALITY_GESTURE)
        if payload is None:
            return

        val = payload['value']
        now = payload['timestamp']

        if val == self.last_gesture_val and (now - self.last_gesture_ts) < GESTURE_DEBOUNCE_SEC:
            self.get_logger().debug(f'[去抖] 手势重复指令忽略: {val}')
            return

        self.last_gesture_val = val
        self.last_gesture_ts  = now
        self._handle_input(payload)

    def _on_eye_target(self, msg: String):
        payload = self._parse_eye_target(msg.data)
        if payload is None:
            return

        target = payload['value']
        # ★ FIX Bug3: 去抖使用本地时钟，避免发布方/接收方时钟不同步问题
        now = time.time()

        if target == self.last_eye_target and (now - self.last_eye_ts) < EYE_TARGET_DEBOUNCE_SEC:
            self.get_logger().debug(f'[去抖] 眼动目标重复忽略: {target}')
            return

        old_target = self.current_target
        self.last_eye_target  = target
        self.last_eye_ts      = now
        self.current_target   = target
        self.current_target_ts = now

        self.get_logger().info(f'👁️ [目标切换] {old_target} → {target}')

        # 目标切换：向旧目标发送 STOP，重置方向状态机
        if old_target != target and old_target in (DOG_ACTIVE_TARGETS | VEHICLE_ACTIVE_TARGETS):
            old_dir = self.active_direction
            self.active_direction    = None
            self.active_direction_ts = 0.0
            self.active_modality     = None
            self.active_confidence   = 0.0

            reason = f'目标切换_{old_target}→{target}_紧急停止'
            if old_dir:
                reason = f'目标切换_{old_target}→{target}_中断_{old_dir}'

            self._publish_command(
                cmd_type='STATE',
                value='STOP',
                control_semantics='DISCRETE',
                decision_reason=reason,
                source_modalities=[MODALITY_EYE],
                target_device=old_target,
            )

        self._publish_feedback_only(
            cmd_type='TARGET',
            value=target,
            control_semantics='DISCRETE',
            decision_reason=f'眼动锁定目标_{target}',
            source_modalities=[MODALITY_EYE],
        )

    # ================================================================
    #  输入解析
    # ================================================================
    def _parse_input(self, raw: str, default_modality: str) -> Optional[Dict[str, Any]]:
        raw = raw.strip()
        if not raw:
            return None

        if raw.startswith('{'):
            try:
                data  = json.loads(raw)
                value = str(data.get('value', '')).strip().upper()
                if not value:
                    self.get_logger().warn(f'JSON 缺少 value 字段: {raw}')
                    return None
                return {
                    'value'            : value,
                    'modality'         : data.get('modality', default_modality),
                    'type'             : data.get('type', self._infer_type(value)),
                    'control_semantics': data.get('control_semantics', self._infer_semantics(value)),
                    'confidence'       : float(data.get('confidence', 0.9)),
                    'timestamp'        : float(data.get('timestamp', time.time())),
                }
            except json.JSONDecodeError:
                self.get_logger().warn(f'JSON 解析失败，尝试裸字符串处理: {raw}')

        value = raw.upper()
        return {
            'value'            : value,
            'modality'         : default_modality,
            'type'             : self._infer_type(value),
            'control_semantics': self._infer_semantics(value),
            'confidence'       : 0.8,
            'timestamp'        : time.time(),
        }

    def _parse_eye_target(self, raw: str) -> Optional[Dict[str, Any]]:
        raw = raw.strip()
        if not raw:
            return None

        if raw.startswith('{'):
            try:
                data  = json.loads(raw)
                value = str(data.get('value', '')).strip().upper()
                if value not in TARGET_VALUES:
                    self.get_logger().warn(f'非法 eye target: {value}')
                    return None
                return {
                    'value'            : value,
                    'modality'         : data.get('modality', MODALITY_EYE),
                    'type'             : data.get('type', 'TARGET'),
                    'control_semantics': data.get('control_semantics', 'DISCRETE'),
                    'confidence'       : float(data.get('confidence', 0.95)),
                    'timestamp'        : float(data.get('timestamp', time.time())),
                }
            except Exception as e:
                self.get_logger().warn(f'eye target JSON 解析失败: {e}')
                return None

        value = raw.upper()
        if value not in TARGET_VALUES:
            self.get_logger().warn(f'非法 eye target 裸字符串: {value}')
            return None

        return {
            'value'            : value,
            'modality'         : MODALITY_EYE,
            'type'             : 'TARGET',
            'control_semantics': 'DISCRETE',
            'confidence'       : 0.8,
            'timestamp'        : time.time(),
        }

    def _infer_type(self, value: str) -> str:
        if value in DIRECTION_COMMANDS:
            return 'DIRECTION'
        if value in STOP_COMMANDS:
            return 'DISCRETE'
        if value in DISCRETE_COMMANDS:
            return 'DISCRETE'
        return 'UNKNOWN'

    def _infer_semantics(self, value: str) -> str:
        if value in DIRECTION_COMMANDS:
            return 'CONTINUOUS'
        return 'DISCRETE'

    # ================================================================
    #  核心处理管道
    # ================================================================
    def _handle_input(self, payload: Dict[str, Any]):
        val      = payload['value']
        modality = payload['modality']
        conf     = payload['confidence']

        self.get_logger().info(
            f'[接收] {modality} → {val} '
            f'(conf={conf:.2f}, sem={payload["control_semantics"]}, target={self.current_target})'
        )

        if val in STOP_COMMANDS:
            self._handle_stop(payload)
            return

        valid_targets = DOG_ACTIVE_TARGETS | VEHICLE_ACTIVE_TARGETS
        if self.current_target not in valid_targets:
            self.get_logger().info(
                f'🚧 [门控] 当前目标={self.current_target}，忽略指令: {modality}:{val}'
            )
            return

        if val in DIRECTION_COMMANDS:
            self._handle_direction(payload)
        elif val in DISCRETE_COMMANDS:
            self._handle_discrete(payload)
        else:
            self.get_logger().warn(f'[未知指令] {val}，忽略')

    def _handle_stop(self, payload: Dict[str, Any]):
        old_dir = self.active_direction
        self.active_direction    = None
        self.active_direction_ts = 0.0
        self.active_modality     = None
        self.active_confidence   = 0.0

        reason = f'STOP_命令_来自_{payload["modality"]}'
        if old_dir:
            reason = f'STOP_中断_{old_dir}_来自_{payload["modality"]}'

        self.get_logger().info(f'🛑 [STOP] 清空方向状态 ({old_dir} → None)')

        self._publish_command(
            cmd_type='STATE',
            value='STOP',
            control_semantics='DISCRETE',
            decision_reason=reason,
            source_modalities=[payload['modality']],
        )

    def _handle_direction(self, payload: Dict[str, Any]):
        new_val      = payload['value']
        new_modality = payload['modality']
        new_conf     = payload['confidence']
        now          = payload['timestamp']

        if self.active_direction is None:
            reason = f'新方向_{new_val}_来自_{new_modality}'
            self._update_direction(new_val, new_modality, new_conf, now, reason)

        elif self.active_direction == new_val:
            self.active_direction_ts = now
            self.active_confidence   = max(self.active_confidence, new_conf)
            self.get_logger().debug(f'[方向刷新] {new_val} 时间戳更新')

        else:
            reason = self._resolve_direction_conflict(payload)
            if reason is not None:
                self._update_direction(new_val, new_modality, new_conf, now, reason)

    def _resolve_direction_conflict(self, new_payload: Dict[str, Any]) -> Optional[str]:
        new_val      = new_payload['value']
        new_modality = new_payload['modality']
        cur_modality = self.active_modality

        new_priority = MODALITY_PRIORITY.get(new_modality, 0)
        cur_priority = MODALITY_PRIORITY.get(cur_modality, 0)

        conflict_summary = (
            f'冲突: {cur_modality}:{self.active_direction} vs {new_modality}:{new_val}'
        )
        self.get_logger().info(f'⚡ [冲突] {conflict_summary}')

        if new_priority > cur_priority:
            reason = f'冲突仲裁_语音覆盖手势_{self.active_direction}→{new_val}'
            self.get_logger().info(f'✅ [仲裁] 语音优先: {reason}')
            return reason
        elif new_priority == cur_priority:
            reason = f'冲突仲裁_最新输入优先_{self.active_direction}→{new_val}'
            self.get_logger().info(f'✅ [仲裁] 时间优先: {reason}')
            return reason
        else:
            self.get_logger().info(
                f'⏸ [仲裁] 保持当前 {self.active_direction}（{cur_modality} 优先于 {new_modality}）'
            )
            return None

    def _update_direction(self, val, modality, confidence, ts, reason):
        old = self.active_direction
        self.active_direction    = val
        self.active_direction_ts = ts
        self.active_modality     = modality
        self.active_confidence   = confidence
        self.get_logger().info(f'🔄 [方向更新] {old} → {val} | 原因: {reason}')

    def _handle_discrete(self, payload: Dict[str, Any]):
        val      = payload['value']
        modality = payload['modality']

        self.get_logger().info(f'🎭 [离散] {val} 来自 {modality}（不影响方向状态）')

        self._publish_command(
            cmd_type='DISCRETE',
            value=val,
            control_semantics='DISCRETE',
            decision_reason=f'离散动作_{val}_来自_{modality}',
            source_modalities=[modality],
        )
        self.pending_discrete = payload

    # ================================================================
    #  T0 周期输出（10Hz）
    # ================================================================
    def _on_timer(self):
        now = time.time()

        if self.active_direction is not None:
            elapsed = now - self.active_direction_ts
            if elapsed > DIRECTION_TIMEOUT_SEC:
                self.get_logger().info(
                    f'⏰ [超时] {self.active_direction} 已 {elapsed:.1f}s 无输入 → STANDBY'
                )
                self.active_direction    = None
                self.active_direction_ts = 0.0
                self.active_modality     = None
                self.active_confidence   = 0.0

        self._emit_state_output()

    def _emit_state_output(self):
        """
        输出规则：
        1. 当前目标为 DOG → 输出到 /dog_command
        2. 当前目标为 VEHICLE → 输出到 /car_command
        3. 当前目标为 NONE → 两端都 STANDBY
        有方向 → 输出方向；无方向 → 输出 STANDBY
        """
        active_targets = DOG_ACTIVE_TARGETS | VEHICLE_ACTIVE_TARGETS

        if self.current_target not in active_targets:
            # NONE 状态：向两个设备各发一次 STANDBY
            for tgt in (DOG_ACTIVE_TARGETS | VEHICLE_ACTIVE_TARGETS):
                self._publish_command(
                    cmd_type='STATE',
                    value='STANDBY',
                    control_semantics='CONTINUOUS',
                    decision_reason=f'当前目标_{self.current_target}_所有设备待机',
                    source_modalities=[MODALITY_EYE],
                    target_device=next(iter({tgt})),
                )
            return

        if self.active_direction:
            self._publish_command(
                cmd_type='DIRECTION',
                value=self.active_direction,
                control_semantics='CONTINUOUS',
                decision_reason=f'持续方向_{self.active_direction}_来自_{self.active_modality}',
                source_modalities=[self.active_modality] if self.active_modality else [],
            )
        else:
            self._publish_command(
                cmd_type='STATE',
                value='STANDBY',
                control_semantics='CONTINUOUS',
                decision_reason='无有效方向_待机',
                source_modalities=[],
            )

    # ================================================================
    #  发布函数
    # ================================================================
    def _publish_command(
        self,
        cmd_type: str,
        value: str,
        control_semantics: str,
        decision_reason: str,
        source_modalities: list,
        target_device: str = None,
    ):
        actual_target = target_device or self.current_target
        payload = {
            'type'             : cmd_type,
            'value'            : value,
            'control_semantics': control_semantics,
            'source_modalities': source_modalities,
            'decision_reason'  : decision_reason,
            'timestamp'        : time.time(),
            'current_target'   : self.current_target,
        }

        json_str = json.dumps(payload, ensure_ascii=False)
        msg      = String()
        msg.data = json_str

        if actual_target in DOG_ACTIVE_TARGETS:
            self.command_pub.publish(msg)
        elif actual_target in VEHICLE_ACTIVE_TARGETS:
            self.car_command_pub.publish(msg)

        self.feedback_pub.publish(msg)

        if value != 'STANDBY':
            self.get_logger().info(
                f'📤 [输出] {value} | 类型:{cmd_type} | 语义:{control_semantics} | '
                f'目标:{actual_target} | 原因:{decision_reason}'
            )

    def _publish_feedback_only(
        self,
        cmd_type: str,
        value: str,
        control_semantics: str,
        decision_reason: str,
        source_modalities: list,
    ):
        payload = {
            'type'             : cmd_type,
            'value'            : value,
            'control_semantics': control_semantics,
            'source_modalities': source_modalities,
            'decision_reason'  : decision_reason,
            'timestamp'        : time.time(),
            'current_target'   : self.current_target,
        }
        msg      = String()
        msg.data = json.dumps(payload, ensure_ascii=False)
        self.feedback_pub.publish(msg)

        self.get_logger().info(
            f'📣 [反馈] {value} | 类型:{cmd_type} | 目标:{self.current_target} | 原因:{decision_reason}'
        )


def main(args=None):
    rclpy.init(args=args)
    node = FusionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('\n🛑 融合节点被用户中断')
    except Exception as e:
        node.get_logger().error(f'节点崩溃: {e}')
        raise
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()