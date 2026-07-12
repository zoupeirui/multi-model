#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
dog_controller.py
机器狗控制节点 —— 接收 fusion_node 下发的 /dog_command（JSON），
解析语义指令后驱动 Unitree 机器狗底层 SDK。

【与 fusion_node 的接口】
  输入话题: /dog_command
  消息格式: JSON String
    {
      "type": "DIRECTION" | "DISCRETE" | "STATE",
      "value": "FORWARD" | "STOP" | "HELLO" | ...,
      "control_semantics": "CONTINUOUS" | "DISCRETE",
      "decision_reason": "...",
      "timestamp": 1234567890.0,
      "current_target": "DOG"
    }

【指令分类（匹配 fusion_node 的 DIRECTION/DISCRETE_COMMANDS）】
  LOCOMOTION（运动）: FORWARD, BACKWARD, LEFT, RIGHT, LEFT_MOVE, RIGHT_MOVE, STOP
  ACTION（动作触发）: HELLO, DANCE, ROLL, TWIST
  SPECIAL（特殊）:    TURN_ON, TURN_OFF

【Unitree SDK 适配说明】
  - 本节点封装了 unitree_legged_sdk 调用接口
  - 如果 SDK 不可用（测试环境），自动切换为 dry-run 模式（仅打印）
  - 运动指令通过 HighCmd 发送，动作指令通过 BmsCmd / 预置动作模式发送
"""

import json
import time
import threading
from typing import Optional, Dict, Any

import rclpy
from rclpy.node import Node
from std_msgs.msg import String

# ---- 尝试导入 Unitree SDK ----
try:
    import sys
    sys.path.append('/opt/unitree_legged_sdk/lib/python/amd64')
    import robot_interface as sdk
    UNITREE_SDK_AVAILABLE = True
except ImportError:
    UNITREE_SDK_AVAILABLE = False

# ======================== 话题配置 ========================
DOG_COMMAND_TOPIC  = '/dog_command'
DOG_FEEDBACK_TOPIC = '/dog_control_feedback'

# ======================== 指令集分类 ========================
LOCOMOTION_CMDS = {'FORWARD', 'BACKWARD', 'LEFT', 'RIGHT', 'LEFT_MOVE', 'RIGHT_MOVE'}
ACTION_CMDS     = {'HELLO', 'DANCE', 'ROLL', 'TWIST'}
STOP_CMDS       = {'STOP'}
SPECIAL_CMDS    = {'TURN_ON', 'TURN_OFF'}

# ======================== 运动参数 ========================
# Unitree HighCmd 速度/角速度参数（单位: m/s, rad/s）
SPEED_CONFIGS = {
    'NORMAL': {'vx': 0.3,  'vy': 0.0,  'vyaw': 0.0},
    'FAST'  : {'vx': 0.6,  'vy': 0.0,  'vyaw': 0.0},
    'SLOW'  : {'vx': 0.15, 'vy': 0.0,  'vyaw': 0.0},
}

LOCO_CMD_PARAMS = {
    'FORWARD'   : {'vx':  0.3,  'vy':  0.0,  'vyaw':  0.0},
    'BACKWARD'  : {'vx': -0.3,  'vy':  0.0,  'vyaw':  0.0},
    'LEFT'      : {'vx':  0.0,  'vy':  0.0,  'vyaw':  0.5},   # 原地左转
    'RIGHT'     : {'vx':  0.0,  'vy':  0.0,  'vyaw': -0.5},   # 原地右转
    'LEFT_MOVE' : {'vx':  0.0,  'vy':  0.3,  'vyaw':  0.0},   # 侧向左平移
    'RIGHT_MOVE': {'vx':  0.0,  'vy': -0.3,  'vyaw':  0.0},   # 侧向右平移
    'STOP'      : {'vx':  0.0,  'vy':  0.0,  'vyaw':  0.0},
}

# Unitree 预置动作编号（mode=2 进入动作模式）
ACTION_MOTION_MAP = {
    'HELLO': 1,    # 挥手（BmsCmd 预置动作）
    'DANCE': 12,   # 跳舞
    'ROLL' : 6,    # 翻滚
    'TWIST': 10,   # 扭腰
}

# 速度修饰词前缀
SPEED_MODIFIER_PREFIX = {'FAST', 'SLOW'}


class DogControlNode(Node):
    """
    机器狗控制节点（融合节点接口版）

    修复点（相对于原补丁说明）：
    1. 完整实现 _on_command，内置 JSON 解析
    2. STANDBY 心跳过滤（不向狗发命令）
    3. 速度修饰词剥离（FAST_/SLOW_ 前缀）
    4. SDK 不可用时自动 dry-run
    """

    def __init__(self):
        super().__init__('dog_control_node')

        # ---- ROS2 参数 ----
        self.declare_parameter('dry_run', not UNITREE_SDK_AVAILABLE)
        self.declare_parameter('sdk_mode', 2)   # Unitree HighLevel mode

        self._dry_run = self.get_parameter('dry_run').value
        self._sdk_mode = self.get_parameter('sdk_mode').value

        # ---- SDK 接口 ----
        self._udp    : Optional[Any] = None
        self._cmd    : Optional[Any] = None
        self._state  : Optional[Any] = None
        self._sdk_lock = threading.Lock()

        # ---- 当前运动状态 ----
        self._current_vx   : float = 0.0
        self._current_vy   : float = 0.0
        self._current_vyaw : float = 0.0
        self._speed_scale  : float = 1.0
        self._in_action    : bool  = False   # 是否正在执行动作（阻断方向输出）

        # ---- 发布者 ----
        self.feedback_pub = self.create_publisher(String, DOG_FEEDBACK_TOPIC, 10)

        # ---- 订阅者 ----
        self.create_subscription(String, DOG_COMMAND_TOPIC, self._on_command, 10)

        # ---- 初始化 SDK ----
        self._init_sdk()

        self.get_logger().info('=' * 60)
        self.get_logger().info('🐕 机器狗控制节点已启动')
        self.get_logger().info(f'   Unitree SDK: {"可用" if UNITREE_SDK_AVAILABLE else "不可用"}')
        self.get_logger().info(f'   Dry-run 模式: {self._dry_run}')
        self.get_logger().info('=' * 60)

    # ================================================================
    #  SDK 初始化
    # ================================================================
    def _init_sdk(self):
        if self._dry_run:
            self.get_logger().warn('⚠️  Dry-run 模式：不实际驱动机器狗')
            return

        try:
            self._udp   = sdk.UDP(self._sdk_mode, 8080, "192.168.123.161", 8082)
            self._cmd   = sdk.HighCmd()
            self._state = sdk.HighState()
            self._udp.InitCmdData(self._cmd)
            self.get_logger().info('✅ Unitree SDK 初始化成功')
        except Exception as e:
            self.get_logger().error(f'❌ SDK 初始化失败: {e}，切换为 dry-run')
            self._dry_run = True

    # ================================================================
    #  指令回调（核心入口）
    # ================================================================
    def _on_command(self, msg: String):
        raw = msg.data.strip()
        self.get_logger().info(f'📩 收到下发指令: {raw[:120]}')

        # ---- JSON 解析 ----
        value, cmd_type, decision_reason = self._parse_fusion_msg(raw)
        if value is None:
            return

        # ---- STANDBY 心跳：直接忽略（不发命令给狗）----
        if value == 'STANDBY':
            return

        self.get_logger().info(
            f'[解析] value={value} type={cmd_type} reason={decision_reason}'
        )

        # ---- 速度修饰词剥离 ----
        speed_scale, base_cmd = self._extract_speed_modifier(value)
        if speed_scale != 1.0:
            self._speed_scale = speed_scale

        # ---- 分发处理 ----
        if base_cmd in STOP_CMDS:
            self._handle_stop(base_cmd, decision_reason)
        elif base_cmd in LOCOMOTION_CMDS:
            self._handle_locomotion(base_cmd)
        elif base_cmd in ACTION_CMDS:
            self._handle_action(base_cmd)
        elif base_cmd in SPECIAL_CMDS:
            self._handle_special(base_cmd)
        else:
            self.get_logger().warn(f'❓ 机器狗无法识别指令: {base_cmd}')

    # ================================================================
    #  解析 fusion_node JSON 消息
    # ================================================================
    def _parse_fusion_msg(self, raw: str):
        """返回 (value, cmd_type, decision_reason) 或 (None, None, None)"""
        if raw.startswith('{'):
            try:
                data  = json.loads(raw)
                value = str(data.get('value', '')).strip().upper()
                if not value:
                    self.get_logger().warn(f'JSON 缺少 value 字段: {raw}')
                    return None, None, None
                return (
                    value,
                    data.get('type', 'UNKNOWN'),
                    data.get('decision_reason', ''),
                )
            except json.JSONDecodeError as e:
                self.get_logger().warn(f'JSON 解析失败，尝试裸字符串处理: {e}')

        # 裸字符串兜底
        value = raw.upper()
        if '_' in value:
            parts = value.split('_', 1)
            if parts[0] in ['FAST', 'SLOW']:
                # 仍视为合法指令，交给 _extract_speed_modifier 处理
                pass
        return value, 'UNKNOWN', ''

    # ================================================================
    #  速度修饰词提取
    # ================================================================
    def _extract_speed_modifier(self, value: str):
        """
        'FAST_FORWARD' → (1.5, 'FORWARD')
        'SLOW_LEFT'    → (0.6, 'LEFT')
        'FORWARD'      → (1.0, 'FORWARD')
        """
        for prefix in SPEED_MODIFIER_PREFIX:
            if value.startswith(f'{prefix}_'):
                base = value[len(prefix) + 1:]
                scale = 1.5 if prefix == 'FAST' else 0.6
                self.get_logger().info(f'⚙️ 速度修饰: {prefix}（scale={scale}）基础指令: {base}')
                return scale, base
        return 1.0, value

    # ================================================================
    #  指令处理器
    # ================================================================
    def _handle_stop(self, cmd: str, reason: str = ''):
        self.get_logger().info(f'🛑 [STOP] 停止运动 | reason: {reason}')
        self._in_action = False
        self._speed_scale = 1.0
        self._send_locomotion(vx=0.0, vy=0.0, vyaw=0.0, label='STOP')

    def _handle_locomotion(self, cmd: str):
        if self._in_action:
            self.get_logger().info(f'⏸ [门控] 正在执行动作，运动指令忽略: {cmd}')
            return

        params = LOCO_CMD_PARAMS.get(cmd, {'vx': 0.0, 'vy': 0.0, 'vyaw': 0.0})
        vx   = params['vx']   * self._speed_scale
        vy   = params['vy']   * self._speed_scale
        vyaw = params['vyaw'] * self._speed_scale

        # 限幅（Unitree HighLevel 速度上限约 ±1.0 m/s）
        vx   = max(-1.0, min(1.0, vx))
        vy   = max(-0.6, min(0.6, vy))
        vyaw = max(-1.5, min(1.5, vyaw))

        self.get_logger().info(
            f'🐾 [运动] {cmd} → vx={vx:.2f} vy={vy:.2f} vyaw={vyaw:.2f}'
        )
        self._send_locomotion(vx, vy, vyaw, label=cmd)

    def _handle_action(self, cmd: str):
        motion_id = ACTION_MOTION_MAP.get(cmd)
        if motion_id is None:
            self.get_logger().warn(f'❓ 动作指令 {cmd} 无对应 motion_id')
            return

        self.get_logger().info(f'🎭 [动作] {cmd} → motion_id={motion_id}')
        self._in_action = True
        self._send_action(motion_id, label=cmd)

        # 简单的动作超时重置（避免卡死）
        def reset_action():
            time.sleep(3.0)
            self._in_action = False
            self.get_logger().info(f'✅ [动作完成] {cmd} 执行结束，恢复运动模式')

        threading.Thread(target=reset_action, daemon=True).start()

    def _handle_special(self, cmd: str):
        self.get_logger().info(f'⚡ [特殊] {cmd}')
        if cmd == 'TURN_ON':
            self._send_power(on=True)
        elif cmd == 'TURN_OFF':
            self._send_power(on=False)

    # ================================================================
    #  底层 SDK 发送函数
    # ================================================================
    def _send_locomotion(self, vx: float, vy: float, vyaw: float, label: str):
        """发送运动指令到 Unitree HighLevel"""
        self._current_vx   = vx
        self._current_vy   = vy
        self._current_vyaw = vyaw

        self._publish_feedback(label, {'vx': vx, 'vy': vy, 'vyaw': vyaw}, success=True)

        if self._dry_run:
            self.get_logger().info(
                f'   [dry-run] HighCmd → vx={vx:.2f} vy={vy:.2f} vyaw={vyaw:.2f}'
            )
            return

        with self._sdk_lock:
            if self._udp is None or self._cmd is None:
                self.get_logger().error('❌ SDK 未初始化')
                return
            try:
                self._cmd.mode     = 2        # HighLevel: velocity control mode
                self._cmd.gaitType = 1        # Trot 步态
                self._cmd.velocity = [vx, vy]
                self._cmd.yawSpeed = vyaw
                self._cmd.footRaiseHeight = 0.08
                self._udp.SetSend(self._cmd)
                self._udp.Send()
            except Exception as e:
                self.get_logger().error(f'❌ SDK 发送失败: {e}')

    def _send_action(self, motion_id: int, label: str):
        """发送预置动作指令"""
        self._publish_feedback(label, {'motion_id': motion_id}, success=True)

        if self._dry_run:
            self.get_logger().info(f'   [dry-run] Action → motion_id={motion_id}')
            return

        with self._sdk_lock:
            if self._udp is None or self._cmd is None:
                return
            try:
                self._cmd.mode     = 2
                self._cmd.gaitType = 0
                self._cmd.velocity = [0.0, 0.0]
                self._cmd.yawSpeed = 0.0
                # Unitree 动作通过 BmsCmd 或 action 字段触发（具体字段依 SDK 版本）
                # 下面使用 reserved 字段传 motion_id（需根据实际 SDK 版本调整）
                self._cmd.reserve  = motion_id
                self._udp.SetSend(self._cmd)
                self._udp.Send()
            except Exception as e:
                self.get_logger().error(f'❌ SDK 动作发送失败: {e}')

    def _send_power(self, on: bool):
        """电源控制（占位，具体实现依硬件接口）"""
        state = 'ON' if on else 'OFF'
        self.get_logger().info(f'⚡ [电源] 切换 → {state}')
        if not self._dry_run:
            # TODO: 接入实际电源控制接口
            self.get_logger().warn('电源控制接口未实现，请根据硬件接入')

    # ================================================================
    #  反馈发布
    # ================================================================
    def _publish_feedback(self, label: str, params: dict, success: bool):
        payload = {
            'node'     : 'dog_control_node',
            'value'    : label,
            'params'   : params,
            'success'  : success,
            'timestamp': time.time(),
        }
        msg      = String()
        msg.data = json.dumps(payload, ensure_ascii=False)
        self.feedback_pub.publish(msg)

    # ================================================================
    #  析构
    # ================================================================
    def destroy_node(self):
        self.get_logger().info('🛑 机器狗节点关闭，发送停止指令')
        self._send_locomotion(0.0, 0.0, 0.0, label='SHUTDOWN_STOP')
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = DogControlNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('\n🛑 机器狗节点被用户中断')
    except Exception as e:
        node.get_logger().error(f'节点崩溃: {e}')
        raise
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
