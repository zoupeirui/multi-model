from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'dog_fusion_control'

setup(
    name=package_name,
    version='1.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name] if os.path.exists('resource/' + package_name) else []),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='zpr',
    maintainer_email='zpr@todo.todo',
    description='多模态融合控制系统',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'voice_node = dog_fusion_control.voice_node:main',
            'gesture_node = dog_fusion_control.gesture_node:main',
            'eye_direction_node = dog_fusion_control.eye_direction_node:main',
            'vision_bridge_node = dog_fusion_control.vision_bridge_node:main',
            'target_fusion_node = dog_fusion_control.target_fusion_node:main',
            'action_fusion_node = dog_fusion_control.action_fusion_node:main',
            'dog_controller = dog_fusion_control.dog_controller:main',
            'car_controller = dog_fusion_control.car_controller:main',
        ],
    },
)
