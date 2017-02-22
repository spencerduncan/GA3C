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

import tensorflow as tf

from Config import Config


def ThreadPredictor(coord, server):
    with server.model.graph.as_default():
        with tf.device('/cpu:0'):
            ids = tf.placeholder(tf.float32, [None])
            states = tf.placeholder(tf.float32,
                [None, Config.IMAGE_HEIGHT, Config.IMAGE_WIDTH, Config.STACKED_FRAMES])
        while not coord.should_stop():
            for i in range(len(server.agents)):
                if server.agents[i].pred_red:
                    print("nes?")
                    server.prediction_q.enqueue(server.agents[i].prediction_q)
         #       else:
         #           print(server.agents[i].pred_red)
            ids, state = server.prediction_q.dequeue_up_to(Config.PREDICTION_BATCH_SIZE)

            try:
                 p, v, _ids = server.model.predict_p_and_v(state, ids)
                 print("nes?")
            except TypeError:
                continue

            for i in range(_ids.shape[0]):
                if ids[i] < len(server.agents):
                        server.agents[ids[i]].wait_q.put((p[i], v[i]))
