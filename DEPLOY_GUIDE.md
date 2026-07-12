# Jetson Nano 2GB 部署 ROS 2 Humble + 多模态机器狗控制系统 — 傻瓜式教程

---

## 目录

1. 准备工作
2. 安装 Docker
3. 拉取 / 构建 ROS 2 Humble Docker 镜像
4. 准备项目文件
5. 启动 Docker 容器
6. 容器内安装 Python 依赖
7. 编译 ROS 2 工作空间
8. 逐个启动节点
9. 调试与测试命令大全
10. 开机自启（可选）
11. 常见问题排查

---

## 1. 准备工作

### 1.1 硬件清单

| 设备 | 说明 |
|------|------|
| Jetson Nano 2GB | 刷好 JetPack 4.6.x（Ubuntu 18.04） |
| MicroSD 卡 | 至少 64GB，推荐 128GB |
| USB 麦克风 | 用于语音识别，确认设备 ID=11 |
| 柔性传感手套 | 通过 WiFi 发送 UDP 数据 |
| 便携路由器 | Jetson 和手套连同一 WiFi |
| 优宝特机器狗 | SDK 接口后续接入 |
| 5V 4A 电源 | Jetson Nano 供电（推荐桶形口供电） |

### 1.2 确认系统版本

```bash
# 登录 Jetson Nano 后执行
cat /etc/nv_tegra_release
# 应显示 R32 (release), REVISION: 7.x 之类的 JetPack 4.6 信息

uname -m
# 应显示 aarch64
```

### 1.3 确认网络连接

```bash
# Jetson 连接路由器 WiFi 后
ip addr show wlan0
# 记下 IP 地址，例如 192.168.1.100

# 测试能否上网（拉镜像需要）
ping -c 3 baidu.com
```

---

## 2. 安装 Docker

JetPack 4.6 通常预装了 Docker，先检查：

```bash
docker --version
# 如果显示版本号（如 Docker version 20.10.x），跳到步骤 2.2
```

如果没有安装：

```bash
sudo apt-get update
sudo apt-get install -y docker.io
sudo systemctl enable docker
sudo systemctl start docker
```

### 2.1 让当前用户免 sudo 使用 Docker

```bash
sudo usermod -aG docker $USER

# 重要：执行完后必须重新登录（或重启）才生效
# 方法一：注销再登录
# 方法二：
sudo reboot
```

重新登录后验证：

```bash
docker run hello-world
# 如果看到 "Hello from Docker!" 说明 Docker 正常
```

### 2.2 安装 NVIDIA Container Runtime

这一步让 Docker 容器可以使用 Jetson 的 GPU：

```bash
# 检查是否已安装
dpkg -l | grep nvidia-container
# 如果有输出，跳过安装

# 如果没有，执行安装
sudo apt-get update
sudo apt-get install -y nvidia-container-runtime

# 配置 Docker 默认使用 nvidia runtime
sudo tee /etc/docker/daemon.json << 'EOF'
{
    "runtimes": {
        "nvidia": {
            "path": "nvidia-container-runtime",
            "runtimeArgs": []
        }
    },
    "default-runtime": "nvidia"
}
EOF

sudo systemctl restart docker
```

---

## 3. 拉取 / 构建 ROS 2 Humble Docker 镜像

### 方案 A：使用 Dusty NV 的预编译镜像（推荐，最省事）

Dusty NV 是 NVIDIA 官方维护的 Jetson 容器集合，包含 ROS 2 Humble 的 ARM64 镜像：

```bash
# 拉取镜像（约 5-8GB，请确保网络畅通和存储空间充足）
sudo docker pull dustynv/ros:humble-ros-base-l4t-r32.7.1

# 如果拉取很慢，可以配置 Docker 镜像加速器：
sudo tee /etc/docker/daemon.json << 'EOF'
{
    "runtimes": {
        "nvidia": {
            "path": "nvidia-container-runtime",
            "runtimeArgs": []
        }
    },
    "default-runtime": "nvidia",
    "registry-mirrors": [
        "https://docker.mirrors.ustc.edu.cn",
        "https://hub-mirror.c.163.com"
    ]
}
EOF
sudo systemctl restart docker

# 再次拉取
sudo docker pull dustynv/ros:humble-ros-base-l4t-r32.7.1
```

验证镜像：

```bash
docker images | grep ros
# 应能看到 dustynv/ros  humble-ros-base-l4t-r32.7.1
```

### 方案 B：自己编写 Dockerfile 构建（备选）

如果方案 A 的镜像找不到或版本不匹配，可以自己构建。在 Jetson 上创建文件：

```bash
mkdir -p ~/ros2_docker && cd ~/ros2_docker
```

创建 `Dockerfile`：

```dockerfile
# 基于 NVIDIA L4T 基础镜像
FROM nvcr.io/nvidia/l4t-base:r32.7.1

ENV DEBIAN_FRONTEND=noninteractive
ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8
ENV ROS_DISTRO=humble

# 安装基础工具
RUN apt-get update && apt-get install -y \
    curl gnupg2 lsb-release software-properties-common \
    build-essential cmake git wget \
    python3-pip python3-dev \
    locales \
    && locale-gen en_US en_US.UTF-8 \
    && update-locale LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8 \
    && rm -rf /var/lib/apt/lists/*

# 安装较新的 Python（ROS 2 Humble 需要 Python >= 3.8）
RUN add-apt-repository ppa:deadsnakes/ppa -y \
    && apt-get update \
    && apt-get install -y python3.8 python3.8-dev python3.8-venv \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.8 1 \
    && python3 -m pip install --upgrade pip setuptools wheel \
    && rm -rf /var/lib/apt/lists/*

# 添加 ROS 2 GPG key 和软件源
RUN curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key \
    | apt-key add - \
    && echo "deb http://packages.ros.org/ros2/ubuntu focal main" \
    > /etc/apt/sources.list.d/ros2.list

# 如果官方源速度慢，可换成清华镜像（取消下面两行注释）:
# RUN sed -i 's|packages.ros.org/ros2/ubuntu|mirrors.tuna.tsinghua.edu.cn/ros2/ubuntu|g' \
#     /etc/apt/sources.list.d/ros2.list

# 安装 ROS 2 Humble 基础包
RUN apt-get update && apt-get install -y \
    ros-humble-ros-base \
    python3-colcon-common-extensions \
    python3-rosdep \
    && rm -rf /var/lib/apt/lists/*

# 初始化 rosdep
RUN rosdep init || true && rosdep update

# 设置 ROS 2 环境变量
RUN echo "source /opt/ros/humble/setup.bash" >> /root/.bashrc

WORKDIR /root/robot_ws
CMD ["bash"]
```

构建镜像（耗时 30 分钟 ~ 2 小时，取决于网速）：

```bash
docker build -t ros2_humble_jetson:latest .
```

> **注意**：方案 B 可能因 Jetson Nano 的 Ubuntu 18.04 与 ROS 2 Humble（原生需要 Ubuntu 22.04）之间的兼容性问题而失败。方案 A 的预编译镜像已处理好这些兼容性，所以强烈推荐方案 A。

---

## 4. 准备项目文件

### 4.1 将项目拷贝到 Jetson

将本项目的 `robot_ws` 文件夹完整复制到 Jetson Nano 的 `~/robot_ws` 目录下。

如果从电脑传到 Jetson（假设 Jetson IP 是 192.168.1.100）：

```bash
# 在你的电脑上执行
scp -r robot_ws/ jetson_user@192.168.1.100:~/robot_ws/
```

或者直接在 Jetson 上创建（用 nano/vim 编辑器手动创建每个文件）：

```bash
# 确认目录结构
tree ~/robot_ws/
# 应显示：
# robot_ws/
# └── src/
#     └── robot_control/
#         ├── package.xml
#         ├── setup.py
#         ├── setup.cfg
#         ├── resource/
#         │   └── robot_control
#         └── robot_control/
#             ├── __init__.py
#             ├── gesture_node.py
#             ├── voice_node.py
#             ├── fusion_node.py
#             └── dog_controller.py
```

### 4.2 准备语音模型文件

将 sherpa_onnx 的 Zipformer 模型放到工作目录下：

```bash
mkdir -p ~/robot_ws/model_zipformer

# 将以下 4 个文件放入 ~/robot_ws/model_zipformer/ 目录：
# - tokens.txt
# - encoder-epoch-99-avg-1.int8.onnx
# - decoder-epoch-99-avg-1.int8.onnx
# - joiner-epoch-99-avg-1.int8.onnx

ls ~/robot_ws/model_zipformer/
# 确认 4 个文件都在
```

### 4.3 准备热词文件

```bash
# 创建热词文件（每行一个热词，可按需修改）
cat > ~/robot_ws/hotwords.txt << 'EOF'
前进
后退
左转
右转
左移
右移
停止
启动
关机
跳舞
你好
翻滚
扭
EOF
```

---

## 5. 启动 Docker 容器

### 5.1 完整启动命令

```bash
docker run -it \
    --runtime nvidia \
    --network host \
    --name ros2_robot \
    -e ROS_DOMAIN_ID=42 \
    -e RMW_IMPLEMENTATION=rmw_fastrtps_cpp \
    -v ~/robot_ws:/root/robot_ws \
    -v /home/zpr/glove_ai:/home/zpr/glove_ai:ro \
    -v /dev/snd:/dev/snd \
    --device /dev/snd \
    --privileged \
    -w /root/robot_ws \
    dustynv/ros:humble-ros-base-l4t-r32.7.1 \
    bash
```

**参数逐条解释：**

| 参数 | 作用 |
|------|------|
| `--runtime nvidia` | 使用 NVIDIA 容器运行时，支持 GPU |
| `--network host` | 容器直接使用宿主机网络（UDP 9999 端口可直接接收手套数据） |
| `--name ros2_robot` | 给容器起名，方便后续操作 |
| `-e ROS_DOMAIN_ID=42` | ROS 2 域 ID，同域才能互相发现（数字可自选 0-232） |
| `-e RMW_IMPLEMENTATION=...` | 指定 DDS 中间件实现 |
| `-v ~/robot_ws:/root/robot_ws` | 挂载项目目录到容器内 |
| `-v /home/zpr/glove_ai:...` | 挂载 TTS 模块目录（只读） |
| `-v /dev/snd:/dev/snd` | 挂载声卡设备（麦克风） |
| `--device /dev/snd` | 授权容器访问声卡 |
| `--privileged` | 允许访问所有设备（含音频设备） |
| `-w /root/robot_ws` | 容器内工作目录 |

### 5.2 后续重新进入容器

```bash
# 如果容器已停止
docker start ros2_robot
docker exec -it ros2_robot bash

# 如果容器正在运行，开新终端窗口
docker exec -it ros2_robot bash

# 每次进入容器都要 source 环境
source /opt/ros/humble/setup.bash
source /root/robot_ws/install/setup.bash  # 编译后才有这个文件
```

---

## 6. 容器内安装 Python 依赖

进入容器后执行：

```bash
# 更新 pip
pip3 install --upgrade pip

# 安装语音识别依赖
pip3 install sherpa-onnx numpy sounddevice

# 安装 onnxruntime（CPU 版，适配 aarch64）
pip3 install onnxruntime

# 验证安装
python3 -c "import sherpa_onnx; print('sherpa_onnx OK')"
python3 -c "import sounddevice; print(sounddevice.query_devices())"
python3 -c "import numpy; print('numpy', numpy.__version__)"
```

### 6.1 确认麦克风设备 ID

```bash
python3 -c "
import sounddevice as sd
print(sd.query_devices())
"
# 找到你的 USB 麦克风，确认其序号是 11
# 如果不是 11，需要修改 voice_node.py 中的 MIC_DEVICE_ID
```

---

## 7. 编译 ROS 2 工作空间

在容器内执行：

```bash
# 确保在工作空间根目录
cd /root/robot_ws

# source ROS 2 环境
source /opt/ros/humble/setup.bash

# 编译（使用 --symlink-install 方便调试，修改代码不用重新编译）
colcon build --symlink-install

# 编译成功后 source 安装环境
source install/setup.bash

# 验证节点已注册
ros2 pkg list | grep robot_control
# 应输出: robot_control

ros2 pkg executables robot_control
# 应输出:
# robot_control dog_controller
# robot_control fusion_node
# robot_control gesture_node
# robot_control voice_node
```

> **提示**：如果修改了 Python 代码后需要重新编译，再次执行 `colcon build --symlink-install` 即可。使用 `--symlink-install` 后，大部分修改会即时生效，无需重新编译。

---

## 8. 逐个启动节点

**需要打开 4 个终端窗口**（每个节点一个终端），每个终端都先执行：

```bash
docker exec -it ros2_robot bash
source /opt/ros/humble/setup.bash
source /root/robot_ws/install/setup.bash
export ROS_DOMAIN_ID=42
```

### 终端 1：启动手势节点

```bash
ros2 run robot_control gesture_node
```

### 终端 2：启动语音节点

```bash
cd /root/robot_ws  # 语音模型路径是相对路径，需要在工作空间根目录启动
ros2 run robot_control voice_node
```

### 终端 3：启动融合节点

```bash
ros2 run robot_control fusion_node
```

### 终端 4：启动机器狗控制节点

```bash
ros2 run robot_control dog_controller
```

### 启动顺序建议

推荐按以下顺序启动：`fusion_node` → `dog_controller` → `gesture_node` → `voice_node`

这样融合节点和控制节点先就位，不会漏掉输入节点发出的第一条指令。

---

## 9. 调试与测试命令大全

### 9.1 查看 Topic 列表

```bash
ros2 topic list
# 应显示：
# /gesture_command
# /voice_command
# /dog_command
```

### 9.2 监听各 Topic（实时查看消息）

```bash
# 监听手势指令
ros2 topic echo /gesture_command

# 监听语音指令
ros2 topic echo /voice_command

# 监听融合后的指令
ros2 topic echo /dog_command
```

### 9.3 手动发布测试指令

```bash
# 模拟手势发出 FORWARD 指令
ros2 topic pub --once /gesture_command std_msgs/msg/String "data: 'FORWARD'"

# 模拟语音发出 DANCE 指令
ros2 topic pub --once /voice_command std_msgs/msg/String "data: 'DANCE'"

# 直接向机器狗发送指令（跳过融合）
ros2 topic pub --once /dog_command std_msgs/msg/String "data: 'HELLO'"

# 持续以 1Hz 发送 FORWARD（用于测试运动连续性）
ros2 topic pub -r 1 /gesture_command std_msgs/msg/String "data: 'FORWARD'"
```

### 9.4 查看 Topic 发布频率

```bash
ros2 topic hz /gesture_command
ros2 topic hz /dog_command
```

### 9.5 查看节点信息

```bash
# 列出所有运行中的节点
ros2 node list

# 查看某个节点的详细信息
ros2 node info /gesture_node
ros2 node info /fusion_node
```

### 9.6 验证手套 UDP 连通性

**方法一：在 Jetson 上直接用 Python 收包测试**

```bash
# 在 Jetson 终端中执行（容器内或容器外均可，因为用了 --network host）
python3 -c "
import socket
s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
s.bind(('0.0.0.0', 9999))
print('等待 UDP 数据（端口 9999）...')
while True:
    data, addr = s.recvfrom(1024)
    print(f'收到: {data.decode(\"utf-8\")} 来自 {addr}')
"
```

> 注意：运行此测试时需要先停止 gesture_node，因为端口 9999 不能同时被两个程序占用。

**方法二：从另一台电脑模拟手套发包**

```bash
# 将 192.168.1.100 替换为 Jetson 的实际 IP
python3 -c "
import socket
s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
s.sendto('前进'.encode('utf-8'), ('192.168.1.100', 9999))
print('已发送: 前进')
"
```

**方法三：在 Jetson 本机环回测试**

```bash
# 先启动 gesture_node，然后在另一个终端发送
python3 -c "
import socket
s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
for cmd in ['前进', '停止', '跳舞', '你好']:
    s.sendto(cmd.encode('utf-8'), ('127.0.0.1', 9999))
    print(f'已发送: {cmd}')
    import time; time.sleep(0.5)
"
```

### 9.7 ROS 2 计算图可视化

```bash
# 安装 rqt（如果容器内没有）
apt-get update && apt-get install -y ros-humble-rqt ros-humble-rqt-graph

# 启动可视化
rqt_graph
```

---

## 10. 开机自启（可选）

### 10.1 创建启动脚本

```bash
cat > ~/start_robot.sh << 'SCRIPT'
#!/bin/bash
# 机器狗控制系统一键启动脚本

CONTAINER_NAME="ros2_robot"
WS="/root/robot_ws"
SETUP="source /opt/ros/humble/setup.bash && source ${WS}/install/setup.bash && export ROS_DOMAIN_ID=42"

# 确保容器在运行
docker start ${CONTAINER_NAME} 2>/dev/null

# 在容器内依次启动各节点（后台运行）
docker exec -d ${CONTAINER_NAME} bash -c "${SETUP} && ros2 run robot_control fusion_node"
sleep 1
docker exec -d ${CONTAINER_NAME} bash -c "${SETUP} && ros2 run robot_control dog_controller"
sleep 1
docker exec -d ${CONTAINER_NAME} bash -c "${SETUP} && ros2 run robot_control gesture_node"
sleep 1
docker exec -d ${CONTAINER_NAME} bash -c "${SETUP} && cd ${WS} && ros2 run robot_control voice_node"

echo "所有节点已启动！"
SCRIPT

chmod +x ~/start_robot.sh
```

### 10.2 配置 systemd 开机自启

```bash
sudo tee /etc/systemd/system/robot-control.service << 'EOF'
[Unit]
Description=Robot Dog Control System
After=docker.service
Requires=docker.service

[Service]
Type=oneshot
RemainAfterExit=yes
User=YOUR_USERNAME
ExecStart=/home/YOUR_USERNAME/start_robot.sh
ExecStop=docker stop ros2_robot

[Install]
WantedBy=multi-user.target
EOF

# 替换 YOUR_USERNAME 为实际用户名
sudo sed -i "s/YOUR_USERNAME/$USER/g" /etc/systemd/system/robot-control.service

sudo systemctl daemon-reload
sudo systemctl enable robot-control.service

# 手动测试
sudo systemctl start robot-control.service
sudo systemctl status robot-control.service
```

---

## 11. 常见问题排查

### Q1: Docker 拉取镜像超时

```bash
# 使用国内镜像加速器（已在步骤 3 配置）
# 或者手动导入：在有网络的电脑上 docker save，再 docker load
docker save dustynv/ros:humble-ros-base-l4t-r32.7.1 | gzip > ros2_humble.tar.gz
# 拷贝到 Jetson 后
docker load < ros2_humble.tar.gz
```

### Q2: 端口 9999 被占用

```bash
# 查看谁占用了 9999 端口
ss -tulnp | grep 9999
# 或
lsof -i :9999

# 杀掉占用进程，或修改 gesture_node.py 顶部的 UDP_PORT 常量
```

### Q3: 麦克风无法访问

```bash
# 在容器内检查声卡设备
ls -la /dev/snd/

# 列出音频设备
arecord -l

# 如果看不到设备，确认 docker run 时加了 --device /dev/snd 和 --privileged
```

### Q4: colcon build 报错

```bash
# 常见原因：没有 source ROS 2 环境
source /opt/ros/humble/setup.bash

# 清理后重新编译
rm -rf build/ install/ log/
colcon build --symlink-install
```

### Q5: 节点之间无法通信

```bash
# 确认所有终端的 ROS_DOMAIN_ID 一致
echo $ROS_DOMAIN_ID
# 所有终端都应该是 42

# 确认 RMW 中间件
echo $RMW_IMPLEMENTATION
```

### Q6: sherpa_onnx 模型加载失败

```bash
# 确认模型文件存在
ls -la /root/robot_ws/model_zipformer/

# 确认在正确目录启动 voice_node（模型路径是相对路径）
cd /root/robot_ws
ros2 run robot_control voice_node
```

### Q7: 内存不足（Jetson Nano 2GB）

```bash
# 增加 swap 空间
sudo fallocate -l 4G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
echo '/swapfile swap swap defaults 0 0' | sudo tee -a /etc/fstab

# 关闭桌面环境省内存（如果不需要图形界面）
sudo systemctl set-default multi-user.target
sudo reboot
```

---

## 系统架构总览

```
┌──────────────────────── Jetson Nano 2GB ────────────────────────┐
│                     Docker 容器 (ROS 2 Humble)                   │
│                                                                   │
│  ┌─────────────┐    /gesture_command    ┌──────────────┐         │
│  │ gesture_node │ ──────────────────▶  │              │         │
│  │  (UDP:9999)  │                       │  fusion_node │         │
│  └──────▲───────┘                       │              │         │
│         │ UDP                           │ 手势优先     │         │
│         │                               │ 语音防抖     │         │
│  ┌──────┴───────┐    /voice_command     │              │         │
│  │  柔性手套     │                       │              │         │
│  │ (WiFi/UDP)   │  ┌─────────────┐     │              │         │
│  └──────────────┘  │ voice_node  │ ──▶ │              │         │
│                     │  (麦克风)    │      └──────┬───────┘         │
│                     └─────────────┘             │                 │
│                                        /dog_command               │
│                                                │                  │
│                                      ┌─────────▼──────────┐      │
│                                      │   dog_controller    │      │
│                                      │  (优宝特 SDK 占位)   │      │
│                                      └─────────┬──────────┘      │
│                                                │                  │
└────────────────────────────────────────────────┼──────────────────┘
                                                 │ SDK
                                          ┌──────▼──────┐
                                          │  优宝特机器狗  │
                                          └─────────────┘
```
