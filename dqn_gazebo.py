#!/usr/bin/env python3
#
# Copyright 2019 ROBOTIS CO., LTD.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Authors: Ryan Shim, Gilbert
import rclpy
from rclpy.node import Node
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from gazebo_msgs.srv import DeleteEntity, SpawnEntity
from std_srvs.srv import Empty
from my_msgs.srv import Goal
from geometry_msgs.msg import Pose

import os
import random
import sys

#a node to generate goals
class GazeboInterface(Node):
    def __init__(self, training):
        super().__init__('gazebo_interface')

        if training.lower() == 'false' or training == '0':
            self.training = False
        else:
            self.training = True

        self.consecutive_fails = 0

        # Read the 'Goal' Entity Model
        self.entity_name = 'Goal'
        self.entity = open('./goal_box/model.sdf', 'r').read()

        # initial entity(Goal) position
        self.IndexCounter = 0

        #Initialize clients
        self.delete_entity_client = self.create_client(DeleteEntity, 'delete_entity')
        self.spawn_entity_client = self.create_client(SpawnEntity, 'spawn_entity')
        self.reset_simulation_client = self.create_client(Empty, 'reset_simulation')

        # Initialize services
        self.callback_group = MutuallyExclusiveCallbackGroup()
        self.initialize_env_service = self.create_service(Goal, 'initialize_env', self.initialize_env_callback,callback_group=self.callback_group)
        self.task_succeed_service = self.create_service(Goal, 'task_succeed', self.task_succeed_callback, callback_group=self.callback_group)
        self.task_failed_service = self.create_service(Goal, 'task_failed', self.task_failed_callback,callback_group=self.callback_group)

        print('- (minus) = Episode failed')
        print('# (hash)  = Episode succeeded')
        print('', end='')

    def reset_simulation(self):
        reset_req = Empty.Request()
        self.IndexCounter = 0

        # check connection to the service server
        while not self.reset_simulation_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().warn('service for reset_simulation is not available, waiting ...')

        self.reset_simulation_client.call_async(reset_req)

    def delete_entity(self):
        delete_req = DeleteEntity.Request()
        delete_req.name = self.entity_name

        # check connection to the service server
        while not self.delete_entity_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().warn('service for delete_entity is not available, waiting ...')

        future = self.delete_entity_client.call_async(delete_req)
        rclpy.spin_until_future_complete(self, future)

    def spawn_entity(self):
        entity_pose = Pose()
        entity_pose.position.x = self.entity_pose_x
        entity_pose.position.y = self.entity_pose_y

        spawn_req = SpawnEntity.Request()
        spawn_req.name = self.entity_name
        spawn_req.xml = self.entity
        spawn_req.initial_pose = entity_pose

        # check connection to the service server
        while not self.spawn_entity_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().warn('service for spawn_entity is not available, waiting ...')

        future = self.spawn_entity_client.call_async(spawn_req)
        rclpy.spin_until_future_complete(self, future)

    def task_succeed_callback(self, request, response):
        self.delete_entity()
        self.generate_goal_pose()
        self.spawn_entity()
        response.pose_x = self.entity_pose_x
        response.pose_y = self.entity_pose_y
        response.success = True
        print('#', end='')
        self.consecutive_fails = 0
        return response

    def task_failed_callback(self, request, response):
        self.consecutive_fails = self.consecutive_fails + 1
        self.delete_entity()
        self.reset_simulation()
        self.spawn_entity()

        if not self.training or self.consecutive_fails > 20:
            self.generate_goal_pose()
            self.consecutive_fails = 0
        
        response.pose_x = self.entity_pose_x
        response.pose_y = self.entity_pose_y
        response.success = True
        print('-', end='')
        
        return response

    def initialize_env_callback(self, request, response):
        self.delete_entity()
        self.reset_simulation()
        self.generate_goal_pose()
        self.spawn_entity()
        response.pose_x = self.entity_pose_x
        response.pose_y = self.entity_pose_y
        response.success = True
        print('Environment initialized')
        print('', end='')
        return response

    def generate_goal_pose(self):
        if self.training:
            if IndexCounter > 18:
                generate_random_pose()
            else:
                goal_pose_list = [
                    [0.72, 0],
                    [1.45, 0],
                    [0.63, 0.56],
                    [0.63, -0.49],
                    [0.36, -1.13],
                    [0.36, 1.06],
                    [-0.41, 1.06],
                    [-0.41, -1.13],
                    [0.36, 1.64],
                    [0.36, -1.64],
                    [1.78, 0.46],
                    [1.78, -1.05],
                    [-0.54, -1.67],
                    [-0.54, 1.67],
                    [-1.18, 0.03],
                    [1.7, 1.7],
                    [-1.7, 1.7],
                    [1.7, -1.7],
                    [-1.7, -1.7]
                ]
                self.entity_pose_x = goal_pose_list[self.IndexCounter][0]
                self.entity_pose_y = goal_pose_list[self.IndexCounter][1]
                self.IndexCounter = self.IndexCounter + 1
            
        else:
            goal_pose_list = [[1.7, 1.7], [-1.7, -1.7], [1.7, -1.7], [-1.7, 1.7], [0.0,0.0]]
            self.entity_pose_x = goal_pose_list[self.IndexCounter][0]
            self.entity_pose_y = goal_pose_list[self.IndexCounter][1]
            self.IndexCounter = (self.IndexCounter + 1) % 5

    def generate_random_pose(self):
        self.entity_pose_x = random.randrange(-23, 23) / 10
        self.entity_pose_y = random.randrange(-23, 23) / 10

        while abs(self.entity_pose_x) > 0.8 and abs(self.entity_pose_x) < 1.2:
            self.entity_pose_x = random.randrange(-23, 23) / 10
        
        while abs(self.entity_pose_y) > 0.8 and abs(self.entity_pose_y) < 1.2:
            self.entity_pose_y = random.randrange(-23, 23) / 10

def main(args=sys.argv[1]):
    rclpy.init(args=args)
    gazebo_interface = GazeboInterface(args)
    while True:
        rclpy.spin_once(gazebo_interface, timeout_sec=0.1)

    rclpy.shutdown()

if __name__ == '__main__':
    main()
