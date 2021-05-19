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

import collections
import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam

import json
import numpy
import os
import random
import sys
import time

import rclpy
from rclpy.node import Node

from turtlebot3_msgs.srv import Dqn

tf.config.set_visible_devices([], 'GPU')
# gpus = tf.config.list_physical_devices('GPU')
# tf.config.set_visible_devices(gpus[0], 'GPU')
strategy = tf.distribute.get_strategy() # works on CPU and single GPU


class DQNTest(Node):
    def __init__(self):
        super().__init__('dqn_test')

        """************************************************************
        ** Initialise variables
        ************************************************************"""

        # State size and action size
        self.state_size = 26
        self.action_size = 5

        # DQN hyperparameter
        self.discount_factor = 0.99
        self.learning_rate = 0.00025
        self.epsilon = 1.0
        self.epsilon_decay = 0.99
        self.epsilon_min = 0.05
        self.batch_size = 64
        self.train_start = 64

        # Replay memory
        self.memory = collections.deque(maxlen=1000000)

        # Build model and target model
        self.model = self.create_qnetwork()

        # Load saved models
        self.model.set_weights(load_model('stage1_episode4500.h5', compile=False).get_weights())
        with open('stage1_episode4500.json') as outfile:
            param = json.load(outfile)
            self.epsilon = param.get('epsilon')

        """************************************************************
        ** Initialise ROS clients
        ************************************************************"""
        # Initialise clients
        self.dqn_com_client = self.create_client(Dqn, 'rl_agent_interface')

        """************************************************************
        ** Start process
        ************************************************************"""
        self.process()

    """*******************************************************************************
    ** Callback functions and relevant functions
    *******************************************************************************"""
    def process(self):
        global_step = 0

        for episode in range(1, 1000):
            global_step += 1
            local_step = 0

            state = list()
            next_state = list()
            done = False
            init = True
            score = 0

            # Reset DQN environment
            time.sleep(0.05)

            while not done:
                local_step += 1

                # Aciton based on the current state
                if local_step == 1:
                    action = 2  # Move forward
                else:
                    state = next_state
                    action = int(self.get_action(state))

                # Send action and receive next state and reward
                req = Dqn.Request()
                print(int(action))
                req.action = action
                req.init = init
                while not self.dqn_com_client.wait_for_service(timeout_sec=1.0):
                    self.get_logger().info('service not available, waiting again...')

                future = self.dqn_com_client.call_async(req)

                while rclpy.ok():
                    rclpy.spin_once(self)
                    if future.done():
                        if future.result() is not None:
                            # Next state and reward
                            next_state = future.result().state
                            reward = future.result().reward
                            done = future.result().done
                            score += reward
                            init = False
                        else:
                            self.get_logger().error(
                                'Exception while calling service: {0}'.format(future.exception()))
                        break

                # While loop rate
                time.sleep(0.01)

    def create_qnetwork(self):
        with strategy.scope():
            model = Sequential()
            model.add(Dense(512, input_shape=(self.state_size,), activation='relu'))
            model.add(Dense(256, activation='relu'))
            #model.add(Dense(256, activation='relu'))
            model.add(Dense(128, activation='relu'))
            #model.add(Dense(128, activation='relu'))
            model.add(Dense(64, activation='relu'))
            #model.add(Dense(64, activation='relu'))
            model.add(Dense(32, activation='relu'))
            model.add(Dense(self.action_size, activation='linear'))
            model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
            model.summary()

            return model

    def get_action(self, state):
        state = numpy.asarray(state)
        q_value = self.model.predict(state.reshape(1, len(state)))
        print(numpy.argmax(q_value[0]))
        return numpy.argmax(q_value[0])


rclpy.init()
dqn_test = DQNTest()
rclpy.spin(dqn_test)

dqn_test.destroy()
rclpy.shutdown()

