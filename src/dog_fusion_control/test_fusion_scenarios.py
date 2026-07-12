#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
test_fusion_scenarios.py（已修复版）
融合节点测试脚本 — 覆盖全部 6 个演示场景 + VEHICLE 切换验证

修复点：
1. 删除未定义的 step() / sep() 调用，改为统一的 _section() 函数
2. 新增 /car_command 订阅，切换为 VEHICLE 后也能验证输出
3. spin_for 增加了双通道消息收集显示
4. 删除 #sym:print 残留调试符号

使用方法：
  终端1: ros2 run <pkg> fusion_node_fixed
  终端2: ros2 topic echo /dog_command
  终端3: ros2 topic echo /car_command
  终端4: python3 test_fusion_scenarios.py
"""

import json
import time
import rclpy
from rclpy.node import Node
from std_msgs.msg import String

VOICE_TOPIC   = '/voice_command'
GESTURE_TOPIC = '/gesture_command'
EYE_TOPIC     = '/eye_target'
DOG_TOPIC     = '/dog_command'
CAR_TOPIC     = '/car_command'


def make_payload(modality: str, value: str, confidence: float = 0.95) -> str:
    """构造标准 JSON 指令"""
    type_map = {
        'FORWARD': 'DIRECTION', 'BACKWARD': 'DIRECTION',
        'LEFT': 'DIRECTION',    'RIGHT': 'DIRECTION',
        'LEFT_MOVE': 'DIRECTION', 'RIGHT_MOVE': 'DIRECTION',
        'STOP': 'DISCRETE',    'HELLO': 'DISCRETE',
        'DANCE': 'DISCRETE',   'ROLL': 'DISCRETE', 'TWIST': 'DISCRETE',
    }
    sem_map = {
        'FORWARD': 'CONTINUOUS', 'BACKWARD': 'CONTINUOUS',
        'LEFT': 'CONTINUOUS',    'RIGHT': 'CONTINUOUS',
        'LEFT_MOVE': 'CONTINUOUS', 'RIGHT_MOVE': 'CONTINUOUS',
    }
    payload = {
        'modality'        : modality,
        'type'            : type_map.get(value, 'UNKNOWN'),
        'value'           : value,
        'control_semantics': sem_map.get(value, 'DISCRETE'),
        'confidence'      : confidence,
        'timestamp'       : time.time(),
    }
    return json.dumps(payload, ensure_ascii=False)


def _section(title: str):
    """场景标题打印（替代原来未定义的 step/sep）"""
    print(f'\n{"─" * 55}')
    print(f'  {title}')
    print(f'{"─" * 55}')


class TestNode(Node):
    def __init__(self):
        super().__init__('test_fusion_node')
        self.voice_pub   = self.create_publisher(String, VOICE_TOPIC,   10)
        self.gesture_pub = self.create_publisher(String, GESTURE_TOPIC, 10)
        self.eye_pub     = self.create_publisher(String, EYE_TOPIC,     10)

        # ★ FIX Bug4: 同时订阅 /dog_command 和 /car_command
        self.dog_received : list = []
        self.car_received : list = []
        self.create_subscription(String, DOG_TOPIC, self._on_dog_output, 10)
        self.create_subscription(String, CAR_TOPIC, self._on_car_output, 10)

    def _on_dog_output(self, msg: String):
        try:
            self.dog_received.append(json.loads(msg.data))
        except Exception:
            self.dog_received.append({'raw': msg.data})

    def _on_car_output(self, msg: String):
        try:
            self.car_received.append(json.loads(msg.data))
        except Exception:
            self.car_received.append({'raw': msg.data})

    def pub_eye(self, target: str):
        payload = {
            'modality'        : 'EYE',
            'type'            : 'TARGET',
            'value'           : target,
            'control_semantics': 'DISCRETE',
            'confidence'      : 0.95,
            'timestamp'       : time.time(),
        }
        msg      = String()
        msg.data = json.dumps(payload, ensure_ascii=False)
        self.eye_pub.publish(msg)
        print(f'  → 发布眼动目标: {target}')

    def pub_voice(self, value: str, conf: float = 0.95):
        msg      = String()
        msg.data = make_payload('VOICE', value, conf)
        self.voice_pub.publish(msg)
        print(f'  → 发布语音: {value}')

    def pub_gesture(self, value: str, conf: float = 0.95):
        msg      = String()
        msg.data = make_payload('GESTURE', value, conf)
        self.gesture_pub.publish(msg)
        print(f'  → 发布手势: {value}')

    def spin_for(self, seconds: float, label: str = ''):
        """让 ROS2 spin 若干秒，打印双通道输出摘要"""
        deadline     = time.time() + seconds
        prev_dog_len = len(self.dog_received)
        prev_car_len = len(self.car_received)

        while time.time() < deadline:
            rclpy.spin_once(self, timeout_sec=0.05)

        def dedup(msgs, start):
            values = [m.get('value', '?') for m in msgs[start:]]
            result = []
            for v in values:
                if not result or result[-1] != v:
                    result.append(v)
            return result

        dog_vals = dedup(self.dog_received, prev_dog_len)
        car_vals = dedup(self.car_received, prev_car_len)

        tag = f'[{label}]' if label else f'[{seconds}s]'
        if dog_vals:
            print(f'    {tag} /dog_command → {dog_vals}')
        if car_vals:
            print(f'    {tag} /car_command → {car_vals}')
        if not dog_vals and not car_vals:
            print(f'    {tag} (无新输出)')


# ====================================================================
#  场景执行
# ====================================================================
def run_scenarios(node: TestNode):

    # 注意：fusion_node_fixed 默认 current_target = DOG
    # 因此无需在测试开始时发布眼动，可直接开始手势测试。
    # 以下仍保留眼动发布以验证切换逻辑。

    _section('初始化：激活眼动目标 → DOG')
    node.pub_eye('DOG')
    node.spin_for(1.0, '等待眼动生效')

    # ------------------------------------------------------------------
    # 场景1: 手势持续前进
    # 预期: /dog_command 持续收到 FORWARD（10Hz，去重后显示1个）
    # ------------------------------------------------------------------
    _section('场景1: 手势持续前进')
    print('预期: /dog_command 持续收到 FORWARD')
    node.pub_gesture('FORWARD')
    node.spin_for(1.5, '持续前进')

    # ------------------------------------------------------------------
    # 场景2: 语音打断（STOP）
    # 预期: FORWARD → 立即变 STOP → STANDBY
    # ------------------------------------------------------------------
    _section('场景2: 语音打断 STOP')
    node.pub_gesture('FORWARD')
    node.spin_for(0.3, 'FORWARD起步')
    print('预期: 收到 STOP，然后 STANDBY')
    node.pub_voice('STOP')
    node.spin_for(1.0, 'STOP后状态')

    # ------------------------------------------------------------------
    # 场景3: 方向实时切换
    # 预期: FORWARD → LEFT（立即切换，无中间停顿）
    # ------------------------------------------------------------------
    _section('场景3: 方向实时切换 FORWARD → LEFT')
    node.pub_gesture('FORWARD')
    node.spin_for(0.5, 'FORWARD阶段')
    print('预期: 切换为 LEFT')
    node.pub_gesture('LEFT')
    node.spin_for(0.8, 'LEFT阶段')
    node.pub_voice('STOP')
    node.spin_for(0.3, '清理')

    # ------------------------------------------------------------------
    # 场景4: 离散动作（不影响方向）
    # 预期: FORWARD 持续 + HELLO 单次 → 继续 FORWARD
    # ------------------------------------------------------------------
    _section('场景4: 离散动作 HELLO（不干扰 FORWARD）')
    node.pub_gesture('FORWARD')
    node.spin_for(0.4, 'FORWARD中')
    print('预期: 发出 HELLO 一次，然后继续 FORWARD')
    node.pub_voice('HELLO')
    node.spin_for(1.0, 'HELLO后')
    node.pub_voice('STOP')
    node.spin_for(0.3, '清理')

    # ------------------------------------------------------------------
    # 场景5: 超时自动 STANDBY
    # 预期: 1s 无输入后从 FORWARD 变 STANDBY
    # ------------------------------------------------------------------
    _section('场景5: 超时自动 STANDBY')
    node.pub_gesture('FORWARD')
    node.spin_for(0.3, 'FORWARD起始')
    print('预期: 1s 后自动变为 STANDBY（无新手势输入）')
    node.spin_for(1.8, '等待超时')

    # ------------------------------------------------------------------
    # 场景6: 模态冲突（语音优先）
    # 预期: 手势 FORWARD + 语音 LEFT → 输出 LEFT
    # ------------------------------------------------------------------
    _section('场景6: 模态冲突 — 手势FORWARD vs 语音LEFT（语音优先）')
    node.pub_gesture('FORWARD')
    node.spin_for(0.2, '手势FORWARD')
    print('预期: 语音 LEFT 覆盖手势 FORWARD')
    node.pub_voice('LEFT')
    node.spin_for(0.8, '冲突后输出')
    node.pub_voice('STOP')
    node.spin_for(0.3, '清理')

    # ------------------------------------------------------------------
    # 场景7: 切换至 VEHICLE（★ 新增，原脚本未完成）
    # 预期: /dog_command 收到 STOP，/car_command 开始输出 FORWARD
    # ------------------------------------------------------------------
    _section('场景7: 眼动切换 DOG → VEHICLE')
    print('预期: /dog_command 收到 STOP，/car_command 收到 FORWARD')
    node.pub_eye('VEHICLE')
    node.spin_for(1.2, '等待切换去抖(0.8s)')
    node.pub_gesture('FORWARD')
    node.spin_for(1.0, '车控前进')
    node.pub_voice('STOP')
    node.spin_for(0.3, '清理')

    print('\n\n✅ 全部场景执行完毕')
    print(f'   /dog_command 总计收到 {len(node.dog_received)} 条消息')
    print(f'   /car_command 总计收到 {len(node.car_received)} 条消息')


def main():
    rclpy.init()
    node = TestNode()

    print('🔧 等待融合节点就绪（2s）...')
    deadline = time.time() + 2.0
    while time.time() < deadline:
        rclpy.spin_once(node, timeout_sec=0.1)

    try:
        run_scenarios(node)
    except KeyboardInterrupt:
        print('\n中断测试')
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()