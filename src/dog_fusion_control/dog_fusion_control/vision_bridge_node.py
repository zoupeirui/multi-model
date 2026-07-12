#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
vision_bridge_node.py
========================
Bridge node: receives YOLO detection results from board B (via UDP)
and republishes them on ROS2 topic /vision/objects.

【架构定位】
  板B (YOLO ONNX 推理) ──UDP 8889──→ vision_bridge_node ──/vision/objects──→ fusion_node

【输入】
  UDP 端口 8889 (默认)，JSON 格式：
  {
    "modality": "VISION",
    "type": "OBJECT_DETECTION",
    "timestamp": 1747300000.123,
    "image_width": 1280,
    "image_height": 720,
    "objects": [
      {"label": "robot_car",
       "confidence": 0.92,
       "bbox": [x1, y1, x2, y2],
       "x_center": 481,
       "y_center": 359}
    ]
  }

【输出】
  Topic: /vision/objects
  Type:  std_msgs/String (JSON payload, 附加 source_addr 和 recv_time 字段)

【参数】
  udp_host           : 监听地址（默认 0.0.0.0，监听所有网卡）
  udp_port           : UDP 端口（默认 8889）
  poll_hz            : 轮询频率（默认 30 Hz，与 YOLO 推理速度匹配）
  stale_timeout_sec  : 数据陈旧告警阈值（默认 1.0 秒）

【运行方式】
  ros2 run vision_bridge vision_bridge_node
  ros2 run vision_bridge vision_bridge_node --ros-args -p udp_port:=8889
"""

import json
import socket
import time

import rclpy
from rclpy.node import Node
from std_msgs.msg import String

# ======================== 默认配置 ========================
DEFAULT_UDP_HOST = '0.0.0.0'
DEFAULT_UDP_PORT = 8889
DEFAULT_POLL_HZ = 30
DEFAULT_STALE_TIMEOUT_SEC = 1.0
PUBLISH_TOPIC = '/vision/objects'


class VisionBridgeNode(Node):
    """
    监听 UDP 端口接收板 B 的 YOLO 检测结果，
    透传为 ROS2 topic /vision/objects（JSON String）。
    """

    def __init__(self):
        super().__init__('vision_bridge_node')

        # ---- ROS2 参数声明 ----
        self.declare_parameter('udp_host', DEFAULT_UDP_HOST)
        self.declare_parameter('udp_port', DEFAULT_UDP_PORT)
        self.declare_parameter('poll_hz', DEFAULT_POLL_HZ)
        self.declare_parameter('stale_timeout_sec', DEFAULT_STALE_TIMEOUT_SEC)

        host = self.get_parameter('udp_host').value
        port = int(self.get_parameter('udp_port').value)
        poll_hz = float(self.get_parameter('poll_hz').value)
        self.stale_timeout = float(self.get_parameter('stale_timeout_sec').value)

        # ---- UDP socket 初始化 ----
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setblocking(False)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            self.sock.bind((host, port))
        except OSError as e:
            self.get_logger().error(
                f'❌ UDP 端口绑定失败 {host}:{port}: {e}'
            )
            self.get_logger().error(
                '   请检查是否有其他进程占用，'
                '可执行: sudo fuser -k {}/udp'.format(port)
            )
            raise

        # ---- ROS2 发布者 ----
        self.pub = self.create_publisher(String, PUBLISH_TOPIC, 10)

        # ---- 状态统计 ----
        self.recv_count = 0
        self.publish_count = 0
        self.parse_fail_count = 0
        self.last_recv_time = 0.0
        self.last_log_time = 0.0

        # ---- 轮询定时器 ----
        self.timer = self.create_timer(1.0 / poll_hz, self._poll_udp)

        # ---- 启动日志 ----
        self.get_logger().info('=' * 60)
        self.get_logger().info('👁  vision_bridge_node 已启动')
        self.get_logger().info(f'   UDP 监听: {host}:{port}')
        self.get_logger().info(f'   发布话题: {PUBLISH_TOPIC}')
        self.get_logger().info(f'   轮询频率: {poll_hz} Hz')
        self.get_logger().info(f'   陈旧阈值: {self.stale_timeout} s')
        self.get_logger().info('=' * 60)

    # ================================================================
    #  UDP 轮询
    # ================================================================
    def _poll_udp(self):
        """
        非阻塞读 UDP；每次定时器触发尽量多读一些避免积压。
        """
        # 每次轮询最多读 20 包，避免单次过久阻塞 timer
        for _ in range(20):
            try:
                data, addr = self.sock.recvfrom(65535)
            except BlockingIOError:
                break
            except Exception as e:
                self.get_logger().error(f'UDP 读取异常: {e}')
                return

            # ---- 解析 JSON ----
            try:
                payload = json.loads(data.decode('utf-8'))
            except Exception as e:
                self.parse_fail_count += 1
                self.get_logger().warn(
                    f'JSON 解析失败 from {addr}: {e}  '
                    f'(累计 {self.parse_fail_count} 次)'
                )
                continue

            # ---- 注入接收元数据 ----
            payload['source_addr'] = f'{addr[0]}:{addr[1]}'
            payload['recv_time'] = time.time()

            # ---- 转 ROS2 消息并发布 ----
            msg = String()
            msg.data = json.dumps(payload, ensure_ascii=False)
            self.pub.publish(msg)

            self.recv_count += 1
            self.publish_count += 1
            self.last_recv_time = time.time()

        # ---- 周期性状态日志（每 2 秒）----
        self._maybe_log_status()

    # ================================================================
    #  周期性状态日志
    # ================================================================
    def _maybe_log_status(self):
        now = time.time()
        if now - self.last_log_time < 2.0:
            return
        self.last_log_time = now

        if self.last_recv_time > 0:
            staleness = now - self.last_recv_time
            if staleness < self.stale_timeout:
                self.get_logger().info(
                    f'📡 转发统计: 已发布 {self.publish_count} 包，'
                    f'最近接收 {staleness:.2f}s 前'
                )
            else:
                self.get_logger().warn(
                    f'⚠️  UDP 数据陈旧 {staleness:.1f}s，'
                    '板 B 可能已停止或网络中断'
                )
        else:
            self.get_logger().warn(
                '⚠️  尚未收到任何 UDP 数据，请检查板 B 是否启动 yolo_udp_sender'
            )

    # ================================================================
    #  析构
    # ================================================================
    def destroy_node(self):
        self.get_logger().info(
            f'🛑 vision_bridge_node 关闭，共转发 {self.publish_count} 包'
        )
        try:
            self.sock.close()
        except Exception:
            pass
        super().destroy_node()


# ================================================================
#  main
# ================================================================
def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = VisionBridgeNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        if node:
            node.get_logger().info('vision_bridge_node 被用户中断')
    except Exception as e:
        if node:
            node.get_logger().error(f'节点崩溃: {e}')
        else:
            print(f'节点初始化失败: {e}')
        raise
    finally:
        if node:
            node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
