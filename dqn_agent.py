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
import datetime
import json
import math
import os
import random as rnd
import sys
import time
import numpy as np

import rclpy
from rclpy.node import Node
from turtlebot3_msgs.srv import Dqn
from std_srvs.srv import Empty

import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam

LOGGING = True
current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
dqn_reward_log_dir = 'logs/' + current_time

# print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
# gpus = tf.config.list_physical_devices('GPU')
# tf.config.set_visible_devices(gpus[0], 'GPU')

tf.config.set_visible_devices([], 'GPU')
strategy = tf.distribute.get_strategy() # works on CPU and single GPU

class DQNMetric(tf.keras.metrics.Metric):

    def __init__(self, name='dqn_metric'):
        super(DQNMetric, self).__init__(name=name)
        self.loss = self.add_weight(name='loss', initializer='zeros')
        self.episode_step = self.add_weight(name='step', initializer='zeros')

    def update_state(self, y_true, y_pred=0, sample_weight=None):
        self.loss.assign_add(y_true)
        self.episode_step.assign_add(1)

    def result(self):
        return self.loss / self.episode_step

    def reset_states(self):
        self.loss.assign(0)
        self.episode_step.assign(0)

class DQNAgent(Node):
    def __init__(self, stage):
        super().__init__('dqn_agent')

        # Resume from model file
        self.load_model = False
        self.stage = 1
        self.load_episode = 0

	    # Train mode
        self.train_mode = True
        
        # State size and action size
        self.state_size = 26
        self.action_size = 5
        self.max_training_episodes = 10003

        # DQN hyperparameter
        self.discount_factor = 0.99
        self.learning_rate = 0.00025
        self.epsilon = 1.0
        self.step_counter = 0
        self.epsilon_decay = 0.998
        self.epsilon_min = 0.05
        self.batch_size = 40

        # Replay memory
        self.replay_memory = collections.deque(maxlen=1000000)
        self.min_replay_memory_size = 5000

        # Build model and target model
        self.model = self.create_qnetwork()
        self.target_model = self.create_qnetwork()
        self.update_target_model()
        self.update_target_after = 5000
        self.target_update_after_counter = 0

        # Load saved models
        self.model_path = 'stage' + str(self.stage) + '_episode' + str(self.load_episode) + '.h5'
        self.json_path = self.model_path.replace('.h5','.json')

        if self.load_model:
            self.model.set_weights(load_model(self.model_path).get_weights())
            with open(self.json_path) as outfile:
                param = json.load(outfile)
                self.epsilon = param.get('epsilon')

        # Tensorboard Log
        if LOGGING:
            self.dqn_reward_writer = tf.summary.create_file_writer(dqn_reward_log_dir)
            self.dqn_reward_metric = DQNMetric()

        # Initialise clients
        self.rl_agent_interface_client = self.create_client(Dqn, 'rl_agent_interface')
        self.make_environment_client = self.create_client(Empty, 'make_environment')
        self.reset_environment_client = self.create_client(Dqn, 'reset_environment')

        self.process()

    def process(self):
        
        #environment init
        while not self.make_environment_client.wait_for_service(timeout_sec=1.0):
            print('Environment client ...')
        self.make_environment_client.call_async(Empty.Request())

        time.sleep(1.0)

        episode_num = 0

        for episode in range(self.load_episode + 1, self.max_training_episodes):
            episode_num += 1
            local_step = 0
            score = 0

            # Reset DQN environment
            state = self.reset_environment()
            time.sleep(1.0)

            while True:
                local_step += 1
                action = int(self.get_action(state))

                next_state, reward, done = self.step(action)
                #print('reward ',reward, ' score ',score)
                score += reward

                if self.train_mode:
                    self.append_sample((state, action, reward, next_state, done))
                    self.train_model(done)

                state = next_state
                if done:
                    if LOGGING:
                        self.dqn_reward_metric.update_state(score)
                        with self.dqn_reward_writer.as_default():
                            tf.summary.scalar('dqn_reward', self.dqn_reward_metric.result(), step=episode_num)
                        self.dqn_reward_metric.reset_states()

                    print(
                        "Episode:", episode, "\t",
                        "score:", round(score, 2), "   \t",
                        "memory length:", len(self.replay_memory), "\t",
                        "epsilon:", round(self.epsilon, 3)
                        )

                    param_keys = ['epsilon']
                    param_values = [self.epsilon]
                    param_dictionary = dict(zip(param_keys, param_values))
                    break

                # While loop rate
                time.sleep(0.01)

            # Update result and save model every 100 episodes
            if self.train_mode:
                if episode % 100 == 0:
                    self.model_path = 'stage' + str(self.stage) + '_episode' + str(episode) + '.h5'
                    self.json_path = self.model_path.replace('.h5','.json')

                    self.model.save(self.model_path)
                    with open(self.json_path, 'w') as outfile:
                        json.dump(param_dictionary, outfile)

            #EPSILON CALCULATION
            #self.epsilon = self.epsilon_min + (1.0 - self.epsilon_min) * math.exp(-1.0 * self.step_counter / self.epsilon_decay)
            self.epsilon *= self.epsilon_decay
            if (self.epsilon < self.epsilon_min):
                self.epsilon = self.epsilon_min   

    def reset_environment(self):
        while not self.reset_environment_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().warn('Reset environment client failed to connect to the server, try again ...')

        future = self.reset_environment_client.call_async(Dqn.Request())

        rclpy.spin_until_future_complete(self, future)
        if future.result() is not None:
            state = future.result().state
            state = np.reshape(np.asarray(state), [1, self.state_size])
        else:
            self.get_logger().error(
                'Exception while calling service: {0}'.format(future.exception()))

        return state

    def step(self, action):
        # Send action and receive next state and reward
        req = Dqn.Request()
        req.action = action

        while not self.rl_agent_interface_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('rl_agent interface service not available, waiting again...')

        future = self.rl_agent_interface_client.call_async(req)

        rclpy.spin_until_future_complete(self, future)

        if future.result() is not None:
            # Next state and reward
            next_state = future.result().state
            next_state = np.reshape(np.asarray(next_state), [1, self.state_size])
            reward = future.result().reward
            done = future.result().done
        else:
            self.get_logger().error(
                'Exception while calling service: {0}'.format(future.exception()))
        return next_state, reward, done

    def create_qnetwork(self):
        with strategy.scope():
            model = Sequential()
            model.add(Dense(64,input_shape=(self.state_size,), activation='relu'))
            model.add(Dense(64,activation='relu'))
            model.add(Dropout(0.2))
            model.add(Dense(self.action_size, activation='linear'))
            model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
            model.summary()

            return model

    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())
        self.target_update_after_counter = 0
        print("*Target model updated*")

    def get_action(self, state):
        if self.train_mode:
            self.step_counter += 1

            #MOVED EPSILON CALC FROM HERE TO PER EPISODE

            lucky = rnd.random()
            if lucky > (1 - self.epsilon):
                return rnd.randint(0, self.action_size - 1)
            else:
                return np.argmax(self.model.predict(state))
        else:
            return np.argmax(self.model.predict(state))

    def append_sample(self, transition):
        self.replay_memory.append(transition)

    def train_model(self, terminal):
        if len(self.replay_memory) < self.min_replay_memory_size:
            return
        data_in_mini_batch = rnd.sample(self.replay_memory, self.batch_size)

        current_states = np.array([transition[0] for transition in data_in_mini_batch])
        current_states = current_states.squeeze()
        current_qvalues_list = self.model.predict(current_states)

        next_states = np.array([transition[3] for transition in data_in_mini_batch])
        next_states = next_states.squeeze()
        next_qvalues_list = self.target_model.predict(next_states)

        x_train = []
        y_train = []

        for index, (current_state, action, reward, next_state, done) in enumerate(data_in_mini_batch):
            if not done:
                future_reward = np.max(next_qvalues_list[index])
                desired_q = reward + self.discount_factor * future_reward
            else:
                desired_q = reward

            current_q_values = current_qvalues_list[index]
            current_q_values[action] = desired_q

            x_train.append(current_state)
            y_train.append(current_q_values)

        x_train = np.array(x_train)
        y_train = np.array(y_train)
        x_train = np.reshape(x_train, [len(data_in_mini_batch), self.state_size])
        y_train = np.reshape(y_train, [len(data_in_mini_batch), self.action_size])

        self.model.fit(tf.convert_to_tensor(x_train, tf.float32), tf.convert_to_tensor(y_train, tf.float32),batch_size=self.batch_size, verbose=0)
        self.target_update_after_counter += 1

        if self.target_update_after_counter > self.update_target_after and terminal:
            self.update_target_model()


def main(args=sys.argv[1]):
    rclpy.init(args=args)
    dqn_agent = DQNAgent(args)
    rclpy.spin(dqn_agent)

    dqn_agent.destroy()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
