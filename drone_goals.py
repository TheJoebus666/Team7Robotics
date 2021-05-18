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
class GoalGenerator(Node):
    def __init__(self):
        super().__init__('gazebo_interface')

        # Read the 'Goal' Entity Model
        self.entity_name = 'Goal'
        self.entity = open('./goal_box/model.sdf', 'r').read()

        #Initialize clients
        self.delete_entity_client = self.create_client(DeleteEntity, 'delete_entity')
        self.spawn_entity_client = self.create_client(SpawnEntity, 'spawn_entity')

        # Initialize services
        self.callback_group = MutuallyExclusiveCallbackGroup()
        self.initialize_env_service = self.create_service(Goal, 'initialize_env', self.dummy_callback,callback_group=self.callback_group)
        self.task_succeed_service = self.create_service(Goal, 'task_succeed', self.dummy_callback, callback_group=self.callback_group)
        self.task_failed_service = self.create_service(Goal, 'task_failed', self.dummy_callback,callback_group=self.callback_group)

    def dummy_callback(self, request, response):
        return response

    def delete_entity(self):
        delete_req = DeleteEntity.Request()
        delete_req.name = self.entity_name

        # check connection to the service server
        while not self.delete_entity_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().warn('service for delete_entity is not available, waiting ...')

        future = self.delete_entity_client.call_async(delete_req)
        rclpy.spin_until_future_complete(self, future)

    def spawn_entity(self):
        self.delete_entity()
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

    def generate_goal_pose(self, corner_number):
        goal_pose_list = [[1.7, 1.7], [-1.7, 1.7], [-1.7, -1.7], [1.7, -1.7]]
        self.entity_pose_x = goal_pose_list[corner_number - 1][0]
        self.entity_pose_y = goal_pose_list[corner_number - 1][1]
        self.spawn_entity()