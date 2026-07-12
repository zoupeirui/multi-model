# 多模态融合控制系统 (Multi-Modal Fusion Control System)

![ROS2](https://img.shields.io/badge/ROS2-Humble-blue.svg)
![Docker](https://img.shields.io/badge/Docker-Supported-blue.svg)
![Python](https://img.shields.io/badge/Python-3.10-blue.svg)

本项目是一个工业级多模态机器狗/机器人控制系统，基于 **ROS 2 Humble** 框架进行深度工程化与规范化重构。系统创新性地实现了 **眼动方向 (Eye Tracking)** + **语音 (Voice)** + **手势 (Gesture)** 的三模态智能融合。

通过将散乱的控制脚本高度封装至统一的 `dog_fusion_control` 核心包中，本系统兼顾了强大的功能性与极高的开发、维护便利性。

---

## 🌟 核心特性

- **三模态智能融合**：
  - **眼动 + 视觉 (Target Fusion)**：结合用户的眼球凝视方向与 YOLO 目标检测，智能锁定场景中的控制对象（机器狗或小车）。
  - **语音 + 手势 (Action Fusion)**：结合 Sherpa-ONNX 语音识别引擎与视觉手势，动态解析动作意图，输出最终动作控制指令。
- **高内聚单包架构**：摒弃传统的碎片化包结构，将感知层、融合层、执行层统一集成在 `dog_fusion_control` 包中，大幅降低编译复杂度和节点通讯开销。
- **容器化一键部署**：提供标准的 Docker & Docker-Compose 支持，一次构建，环境免配，开箱即用。

---

## 🏗️ 系统架构设计

系统的工程逻辑被严格划分为三个层次，实现了数据流的单向解耦流转：

```mermaid
graph TD
    subgraph 感知层 (Perception Layer)
        E[Eye Direction Node] -->|/eye/direction| TF(Target Fusion Node)
        V[Vision Bridge Node] -->|/vision/objects| TF
        S[Voice Node] -->|/voice_command| AF(Action Fusion Node)
        G[Gesture Node] -->|/gesture_command| AF
    end

    subgraph 融合层 (Fusion Layer)
        TF -->|/eye_target <br> (DOG/CAR)| AF
    end

    subgraph 执行层 (Execution Layer)
        AF -->|/dog_command| D[Dog Controller]
        AF -->|/car_command| C[Car Controller]
    end
```

### 节点功能详述

#### 1. 感知层 (Perception)
- `voice_node`：语音控制引擎，动态加载本地 `model_zipformer` (Sherpa-ONNX 离线大模型)，并结合 `hotwords.txt` 识别“前进”、“跳舞”等控制语义。
- `gesture_node`：手势识别节点，处理摄像头或视觉传感器传入的手势状态。
- `eye_direction_node`：眼动追踪节点，分析摄像头流，发布实时的用户视线焦点。
- `vision_bridge_node`：高速 UDP 桥接节点，监听边缘计算板 (如 Jetson 等) 推理并回传的 YOLO 物体框 JSON 数据，转为 ROS 2 标准 Topic。

#### 2. 融合层 (Fusion)
- `target_fusion_node` (一级融合)：基于用户的**眼动方向**，与场景中的**YOLO 检测框**进行空间交集运算，智能得出用户此刻正在注视哪个物理实体（例如：Dog 或 Car）。
- `action_fusion_node` (二级融合)：终极决策中枢（Brain）。在得知目标对象的前提下，通过优先级（融合 > 语音 > 手势）仲裁出用户的操作意图，向指定硬件下发指令。

#### 3. 执行层 (Execution)
- `dog_controller`：机器狗底层驱动，将高级指令映射为狗的关节步态逻辑。
- `car_controller`：机器小车底层驱动，实现轮式底盘控制。

---

## 📂 工程拓扑结构

```text
robot_ws/
├── README.md
├── docker/
│   ├── Dockerfile
│   └── docker-compose.yml
├── model_zipformer/                 # 核心 ONNX 语音识别模型库
│   ├── encoder-epoch-99-avg-1.int8.onnx
│   ├── decoder-epoch-99-avg-1.int8.onnx
│   ├── joiner-epoch-99-avg-1.int8.onnx
│   └── tokens.txt
└── src/
    └── dog_fusion_control/          # 唯一核心 ROS2 功能包
        ├── package.xml
        ├── setup.py
        └── dog_fusion_control/
            ├── voice_node.py
            ├── gesture_node.py
            ├── ... (共计 8 个核心 Python 节点)
```

---

## 🚀 快速上手 (Quick Start)

为了避免 macOS 或 Windows 主机上 ROS 2 环境配置的痛苦，本项目强烈推荐基于 **Docker** 运行。

### 1. 构建与启动环境

在终端中进入项目根目录 `robot_ws`：

```bash
# 后台一键构建并启动 ROS 2 容器环境
docker compose -f docker/docker-compose.yml up --build -d
```
> *注：容器在构建期间，会自动执行 `colcon build`，并将环境变量注入 `~/.bashrc` 中。进入容器后无需手动 `source`。*

### 2. 进入交互终端

```bash
docker exec -it dog_fusion_container bash
```

### 3. 运行多模态节点

容器内已实现即插即用，您可以通过标准的 `ros2 run` 命令启动任意组合的节点：

```bash
# 启动两级融合中枢
ros2 run dog_fusion_control target_fusion_node
ros2 run dog_fusion_control action_fusion_node

# 启动各个感知与执行节点（可根据测试需要在新终端独立启动）
ros2 run dog_fusion_control eye_direction_node
ros2 run dog_fusion_control vision_bridge_node
ros2 run dog_fusion_control voice_node
ros2 run dog_fusion_control gesture_node

ros2 run dog_fusion_control dog_controller
ros2 run dog_fusion_control car_controller
```

> **💡 温馨提示 (For Mac Users)**：
> 由于 Docker for Mac 虚拟机的局限，挂载宿主机摄像头或原生麦克风可能受限。如需直接调用本地的 `cv2.VideoCapture`，建议在宿主机（Mac）搭建轻量 Python 环境单独运行感知节点，而将 ROS 2 的融合及控制逻辑置于 Docker 内。

---
*Built with ❤️ for Robotics Research and Industrial AI Control.*
