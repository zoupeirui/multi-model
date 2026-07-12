#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
car_control_node.py
机器车控制节点 —— 接收 fusion_node 下发的 /car_command（JSON），
将语义指令映射为底层串口协议字符串，通过 Serial 发送给小车。

【车控串口协议（从 esp32_control.cpp 提取）】
  运动指令（$DCR:left,right!\n）：
    FORWARD     → $DCR:1000,-1000!\n
    BACKWARD    → $DCR:-1000,1000!\n
    LEFT        → $DCR:-1000,-1000!\n
    RIGHT       → $DCR:1000,1000!\n
    LEFT_MOVE   → $DCR:-600,-1000!\n   (差速左平移近似)
    RIGHT_MOVE  → $DCR:1000,600!\n     (差速右平移近似)
    STOP/STANDBY→ $DCR:0,0!\n

  速度调节（舵机/PWM 通道）：
    ACCELERATE  → #100P1000T0500!#101P2500T0500!\n
    DECELERATE  → #100P1000T0500!#101P1600T0500!\n

  电源控制：
    TURN_ON     → #004P0600T2000!\n
    TURN_OFF    → #004P2400T2000!\n

【串口配置】
  默认 /dev/ttyUSB0，波特率 115200
  可通过 ROS2 参数 serial_port / baud_rate 覆盖

【运行方式】
  ros2 run <your_pkg> car_control_node
  或直接: python3 car_control_node.py
"""

import json
import time
import threading
from typing import Optional

import rclpy
from rclpy.node import Node
from std_msgs.msg import String

try:
    import serial
    SERIAL_AVAILABLE = True
except ImportError:
    SERIAL_AVAILABLE = False

# ======================== 话题配置 ========================
CAR_COMMAND_TOPIC = '/car_command'
CAR_FEEDBACK_TOPIC = '/car_control_feedback'

# ======================== 串口配置默认值 ========================
DEFAULT_SERIAL_PORT = '/dev/ttyUSB0'
DEFAULT_BAUD_RATE   = 115200
DEFAULT_TIMEOUT     = 1.0

# ======================== 速度参数 ========================
SPEED_NORMAL   = 1000   # 标准速度
SPEED_FAST     = 1500   # 快速（乘以倍率）
SPEED_SLOW     = 600    # 慢速

# ======================== 指令映射表 ========================
# 格式：语义指令 → 底层串口字符串
COMMAND_MAP = {
    # --- 运动方向（$DCR:left_motor,right_motor! 正=前转，负=后转）---
    'FORWARD'    : '$DCR:1000,-1000!\n',
    'BACKWARD'   : '$DCR:-1000,1000!\n',
    'LEFT'       : '$DCR:-1000,-1000!\n',   # 原地左转
    'RIGHT'      : '$DCR:1000,1000!\n',     # 原地右转
    'LEFT_MOVE'  : '$DCR:-600,-1000!\n',    # 差速左偏
    'RIGHT_MOVE' : '$DCR:1000,600!\n',      # 差速右偏
    'STOP'       : '$DCR:0,0!\n',
    'STANDBY'    : '$DCR:0,0!\n',           # 融合节点心跳待机，保持静止

    # --- 速度调节（舵机通道 100/101）---
    'ACCELERATE' : '#100P1000T0500!#101P2500T0500!\n',
    'DECELERATE' : '#100P1000T0500!#101P1600T0500!\n',

    # --- 电源控制 ---
    'TURN_ON'    : '#004P0600T2000!\n',
    'TURN_OFF'   : '#004P2400T2000!\n',
}

# 速度修饰词映射（prefix → 速度缩放比例）
SPEED_MODIFIER = {
    'FAST': 1.5,
    'SLOW': 0.6,
}

# 方向指令集（需要考虑速度缩放）
DIRECTION_COMMANDS = {'FORWARD', 'BACKWARD', 'LEFT', 'RIGHT', 'LEFT_MOVE', 'RIGHT_MOVE'}


class CarControlNode(Node):
    """
    机器车控制节点

    订阅 /car_command（fusion_node JSON格式），
    解析语义指令并通过串口发送底层控制字符串。
    """

    def __init__(self):
        super().__init__('car_control_node')

        # ---- ROS2 参数声明 ----
        self.declare_parameter('serial_port', DEFAULT_SERIAL_PORT)
        self.declare_parameter('baud_rate', DEFAULT_BAUD_RATE)
        self.declare_parameter('dry_run', False)   # True=不开串口，仅打印

        self._serial_port_param = self.get_parameter('serial_port').value
        self._baud_rate_param   = self.get_parameter('baud_rate').value
        self._dry_run           = self.get_parameter('dry_run').value

        # ---- 串口对象 ----
        self._serial: Optional[serial.Serial] = None
        self._serial_lock = threading.Lock()

        # ---- 状态 ----
        self._current_speed_scale : float = 1.0   # 当前速度缩放
        self._last_direction       : str   = ''
        self._last_cmd_ts          : float = 0.0

        # ---- 发布者 ----
        self.feedback_pub = self.create_publisher(String, CAR_FEEDBACK_TOPIC, 10)

        # ---- 订阅者 ----
        self.create_subscription(String, CAR_COMMAND_TOPIC, self._on_command, 10)

        # ---- 初始化串口 ----
        self._init_serial()

        self.get_logger().info('=' * 60)
        self.get_logger().info('🚗 车控节点已启动')
        self.get_logger().info(f'   串口: {self._serial_port_param} @ {self._baud_rate_param}')
        self.get_logger().info(f'   Dry-run模式: {self._dry_run}')
        self.get_logger().info('=' * 60)

    # ================================================================
    #  串口初始化
    # ================================================================
    def _init_serial(self):
        if self._dry_run:
            self.get_logger().warn('⚠️  Dry-run 模式：串口不会真实发送')
            return

        if not SERIAL_AVAILABLE:
            self.get_logger().error('❌ pyserial 未安装！运行: pip install pyserial --break-system-packages')
            return

        try:
            self._serial = serial.Serial(
                port     = self._serial_port_param,
                baudrate = self._baud_rate_param,
                timeout  = DEFAULT_TIMEOUT,
            )
            self.get_logger().info(f'✅ 串口已打开: {self._serial_port_param}')
        except serial.SerialException as e:
            self.get_logger().error(f'❌ 串口打开失败: {e}')
            self.get_logger().warn('   切换为 dry-run 模式继续运行（不发送实际指令）')
            self._dry_run = True

    # ================================================================
    #  指令回调
    # ================================================================
    def _on_command(self, msg: String):
        raw = msg.data.strip()
        if not raw:
            return

        # ---- 解析 JSON ----
        value, cmd_type, decision_reason = self._parse_fusion_msg(raw)
        if value is None:
            return

        self.get_logger().info(
            f'📩 [收到] value={value} type={cmd_type} reason={decision_reason}'
        )

        # ---- STANDBY 心跳处理：只在有上一个方向时发停车 ----
        if value == 'STANDBY':
            if self._last_direction and self._last_direction not in ('STOP', 'STANDBY', ''):
                self._send_serial('$DCR:0,0!\n', 'STANDBY')
                self._last_direction = 'STANDBY'
            return

        # ---- 速度修饰词剥离（如 FAST_FORWARD → FORWARD + scale=1.5）----
        speed_scale, base_value = self._extract_speed_modifier(value)
        if speed_scale != 1.0:
            self._current_speed_scale = speed_scale
            self.get_logger().info(f'⚙️ 速度缩放: {speed_scale:.1f}x（来自修饰词）')

        # ---- 映射为底层字符串 ----
        serial_str = self._map_command(base_value, self._current_speed_scale)
        if serial_str is None:
            self.get_logger().warn(f'❓ 未知指令: {base_value}，忽略')
            return

        # ---- 发送 ----
        self._send_serial(serial_str, base_value)

        # 记录上一个方向
        if base_value in DIRECTION_COMMANDS or base_value in ('STOP',):
            self._last_direction = base_value

    # ================================================================
    #  解析 fusion_node JSON 消息
    # ================================================================
    def _parse_fusion_msg(self, raw: str):
        """
        返回 (value, cmd_type, decision_reason) 或 (None, None, None)
        """
        if raw.startswith('{'):
            try:
                data   = json.loads(raw)
                value  = str(data.get('value', '')).strip().upper()
                if not value:
                    self.get_logger().warn(f'JSON 缺少 value: {raw}')
                    return None, None, None
                return (
                    value,
                    data.get('type', 'UNKNOWN'),
                    data.get('decision_reason', ''),
                )
            except json.JSONDecodeError as e:
                self.get_logger().warn(f'JSON 解析失败，尝试裸字符串: {e}')

        # 裸字符串兜底
        value = raw.upper()
        return value, 'UNKNOWN', ''

    # ================================================================
    #  速度修饰词提取
    # ================================================================
    def _extract_speed_modifier(self, value: str):
        """
        e.g. 'FAST_FORWARD' → (1.5, 'FORWARD')
             'FORWARD'      → (1.0, 'FORWARD')
        """
        for prefix, scale in SPEED_MODIFIER.items():
            if value.startswith(f'{prefix}_'):
                base = value[len(prefix) + 1:]
                return scale, base
        return 1.0, value

    # ================================================================
    #  指令映射（含速度缩放）
    # ================================================================
    def _map_command(self, value: str, speed_scale: float = 1.0) -> Optional[str]:
        """
        将语义指令映射为底层串口字符串。
        方向指令支持速度缩放（修改 $DCR 的数值）。
        """
        if value not in COMMAND_MAP:
            return None

        serial_str = COMMAND_MAP[value]

        # 方向指令 + 非标准速度 → 重新计算 $DCR 数值
        if value in DIRECTION_COMMANDS and speed_scale != 1.0:
            serial_str = self._scale_dcr(serial_str, speed_scale)

        return serial_str

    def _scale_dcr(self, dcr_str: str, scale: float) -> str:
        """
        对 $DCR:left,right!\n 中的数值按比例缩放。
        保持符号（方向），限制在 [-2000, 2000] 内。
        e.g. '$DCR:1000,-1000!\n' × 1.5 → '$DCR:1500,-1500!\n'
        """
        try:
            # 提取 "$DCR:" 和 "!" 之间的内容
            inner = dcr_str.strip()  # '$DCR:1000,-1000!\n'
            body  = inner[5:-2]      # '1000,-1000!'  → '1000,-1000'
            body  = body.rstrip('!')
            parts = body.split(',')
            left  = int(int(parts[0]) * scale)
            right = int(int(parts[1]) * scale)

            # 限幅
            left  = max(-2000, min(2000, left))
            right = max(-2000, min(2000, right))

            return f'$DCR:{left},{right}!\n'
        except Exception as e:
            self.get_logger().warn(f'DCR 缩放失败，使用原始值: {e}')
            return dcr_str

    # ================================================================
    #  串口发送
    # ================================================================
    def _send_serial(self, serial_str: str, label: str):
        self.get_logger().info(f'📤 [发送] {label} → "{serial_str.strip()}"')

        if self._dry_run:
            self.get_logger().info(f'   [dry-run] 未实际发送')
            self._publish_feedback(label, serial_str, success=True)
            return

        with self._serial_lock:
            if self._serial is None or not self._serial.is_open:
                self.get_logger().error('❌ 串口未打开，无法发送')
                self._publish_feedback(label, serial_str, success=False)
                return
            try:
                self._serial.write(serial_str.encode('ascii'))
                self._serial.flush()
                self._last_cmd_ts = time.time()
                self._publish_feedback(label, serial_str, success=True)
            except serial.SerialException as e:
                self.get_logger().error(f'❌ 串口发送失败: {e}')
                self._publish_feedback(label, serial_str, success=False)
                # 尝试重连
                self._try_reconnect()

    def _try_reconnect(self):
        self.get_logger().warn('🔄 尝试重新连接串口...')
        try:
            if self._serial:
                self._serial.close()
            self._serial = serial.Serial(
                port     = self._serial_port_param,
                baudrate = self._baud_rate_param,
                timeout  = DEFAULT_TIMEOUT,
            )
            self.get_logger().info(f'✅ 串口重连成功: {self._serial_port_param}')
        except Exception as e:
            self.get_logger().error(f'❌ 串口重连失败: {e}')
            self._serial = None

    # ================================================================
    #  反馈发布
    # ================================================================
    def _publish_feedback(self, label: str, serial_str: str, success: bool):
        payload = {
            'node'      : 'car_control_node',
            'value'     : label,
            'serial_cmd': serial_str.strip(),
            'success'   : success,
            'timestamp' : time.time(),
        }
        msg      = String()
        msg.data = json.dumps(payload, ensure_ascii=False)
        self.feedback_pub.publish(msg)

    # ================================================================
    #  析构
    # ================================================================
    def destroy_node(self):
        # 发送停车指令再关闭
        if self._serial and self._serial.is_open:
            try:
                self._serial.write(b'$DCR:0,0!\n')
                self._serial.flush()
                self.get_logger().info('🛑 关闭前发送停车指令')
                time.sleep(0.1)
                self._serial.close()
            except Exception:
                pass
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = CarControlNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('\n🛑 车控节点被用户中断')
    except Exception as e:
        node.get_logger().error(f'节点崩溃: {e}')
        raise
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
