FROM ros:humble-ros-base

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    python3-pip \
    ros-humble-tmux \ 
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install --no-cache-dir numpy

WORKDIR /robot_ws
COPY ./src /robot_ws/src

RUN . /opt/ros/humble/setup.sh && colcon build

RUN echo "source /opt/ros/humble/setup.bash" >> ~/.bashrc \
    && echo "source /robot_ws/install/setup.bash" >> ~/.bashrc

CMD ["bash"]
