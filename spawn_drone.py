#!/usr/bin/env python3

"""Inject an SDF or URDF file into Gazebo"""

import sys
import transformations
import rclpy
import image_control
from gazebo_msgs.srv import SpawnEntity
from geometry_msgs.msg import Pose
from tello_msgs.srv import TelloAction
from rclpy.node import Node
import time

class TakeOffClient(Node):
    def __init__(self):
        super().__init__('take_off_client')
        self.client = self.create_client(TelloAction, '/drone1/tello_action')
        while not self.client.wait_for_service(timeout_sec = 0.1):
            print('service unavailable, waiting...')
        self.req = TelloAction.Request()

    def send_request(self, cmd):
        self.req.cmd = cmd
        return self.client.call_async(self.req)

def inject(xml: str, initial_pose: Pose):
    """Create a ROS node, and use it to call the SpawnEntity service"""

    rclpy.init()
    node = rclpy.create_node('inject_node')
    client = node.create_client(SpawnEntity, 'spawn_entity')

    if not client.service_is_ready():
        node.get_logger().info('waiting for spawn_entity service...')
        client.wait_for_service()

    request = SpawnEntity.Request()
    request.xml = xml
    request.initial_pose = initial_pose
    future = client.call_async(request)
    rclpy.spin_until_future_complete(node, future)

    if future.result() is not None:
        node.get_logger().info('response: %r' % future.result())
    else:
        raise RuntimeError('exception while calling service: %r' % future.exception())

    node.destroy_node()
    rclpy.shutdown()

f = open('./tello_1.urdf', 'r')

p = Pose()
p.position.x = float(1)
p.position.y = float(0)
p.position.z = float(0)
q = transformations.quaternion_from_euler(0, 0, float(0))
p.orientation.w = q[0]
p.orientation.x = q[1]
p.orientation.y = q[2]
p.orientation.z = q[3]

inject(f.read(), p)

rclpy.init()
take_off_client = TakeOffClient()
service_return = take_off_client.send_request('takeoff')
time.sleep(6.0)

tello_subscriber = image_control.TelloSubscriber()
rclpy.spin_once(tello_subscriber)
time.sleep(1.0)
tello_subscriber.move_backward()
time.sleep(1.0)
tello_subscriber.rotate(2.7)
time.sleep(1.0)
rclpy.spin(tello_subscriber)

input("Press any key to shut down rclpy")
rclpy.shutdown()
