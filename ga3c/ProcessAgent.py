# Copyright (c) 2016, NVIDIA CORPORATION. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NVIDIA CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

from datetime import datetime
from multiprocessing import Process, Queue, Value

import numpy as np
import time
import tensorflow as tf

from Config import Config
from Environment import Environment
from Experience import Experience


@staticmethod
def _accumulate_rewards(experiences, discount_factor, terminal_reward):
    reward_sum = terminal_reward
    for t in reversed(range(0, len(experiences)-1)):
        r = np.clip(experiences[t].reward, Config.REWARD_MIN, Config.REWARD_MAX)
        reward_sum = discount_factor * reward_sum + r
        experiences[t].reward = reward_sum
    return experiences[:-1]

def convert_data(env, experiences):
    x_ = np.array([exp.state for exp in experiences])
    a_ = np.eye(env.get_num_actions())[np.array([exp.action for exp in experiences])].astype(np.float32)
    r_ = np.array([exp.reward for exp in experiences])
    return x_, r_, a_

def run_episode(env, server, actions, id):
    env.reset()
    done = False
    experiences = []

    time_count = 0
    reward_sum = 0.0

    while not done:
        # very first few frames
        if env.current_state is None:
            env.step(0)  # 0 == NOOP
            continue

        with server.model.graph.as_default():
            with tf.device('/cpu:0'):
                server.prediction_q.enqueue([id, env.current_state])
        prediction, value = wait_q.get()

        if Config.PLAY_MODE:
            action = np.argmax(prediction)
        else:
            action = np.random.choice(actions, p=prediction)

        reward, done = env.step(action)
        reward_sum += reward

        exp = Experience(env.previous_state, action, prediction, reward, done)
        experiences.append(exp)

        if done or time_count == Config.TIME_MAX:
            terminal_reward = 0 if done else value

            updated_exps = _accumulate_rewards(experiences, Config.DISCOUNT, terminal_reward)
            if len(updated_exps) == 0:
                yield None, None, None, total_reward
            else:
                x_, r_, a_ = self.convert_data(env, updated_exps)
                yield x_, r_, a_, reward_sum

            # reset the tmax count
            time_count = 0
            # keep the last experience for the next batch
            experiences = [experiences[-1]]
            reward_sum = 0.0

        time_count += 1

def ProcessAgent(coord, server, id):
    prediction_q = []
    pred_red     = False
        
    training_q = ()
    episode_log_q = episode_log_q
        
    env = Environment()
    num_actions = env.get_num_actions()
    actions = np.arange(self.num_actions)
    discount_factor = Config.DISCOUNT
    #one frame at a time
    wait_q = Queue(maxsize=1)

    with server.model.graph.as_default():
        with tf.device('/cpu:0'):
            while not coord.should_stop():
        
                total_reward = 0
                total_length = 0
                for x_, r_, a_, reward_sum in self.run_episode(env, server, actions, id):
                    total_reward += reward_sum
                    if x_ is None:
                        break
                total_length += len(r_) + 1  # +1 for last frame that we drop
                server.training_q.enqueue([x_, r_, a_])
                server.episode_log_q.put((datetime.now(), total_reward, total_length))
        
   
    
