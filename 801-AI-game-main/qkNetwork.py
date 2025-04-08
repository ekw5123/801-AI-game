import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from collections import deque
import random
import numpy as np

###############################################################
#  SECTION 2: Build Q-Network
#  - define the NN architecture and stores the transitions
#  - using the ReplayBuffer class
###############################################################

class QNetwork(tf.keras.Model):
    """
    DQN network to handle sub-state => Q-value. Using table 2 in paper
    and includes the alpha-weighted sub-state input.
      - Input (sub_state_size*sub_state_size) neurons, ReLU
      - Hidden1 (d1) (sub_state_size*sub_state_size) neurons, tanh
      - Hidden2 (d2) (sub_state_size*sub_state_size) neurons, linear
      - Hidden3 (d3) (sub_state_size*sub_state_size) neurons, linear
      - Output = 1 neuron, tanh
    """
    def __init__(self, sub_state_size=3): # this arcitecture is based on table 2 in paper
        super(QNetwork, self).__init__()
        self.flatten = layers.Flatten()
        self.input_layer = layers.Dense(sub_state_size*sub_state_size, activation='relu'        )
        self.d1 = layers.Dense(sub_state_size*sub_state_size, activation='tanh')
        self.d2 = layers.Dense(sub_state_size*sub_state_size, activation='linear')
        self.d3 = layers.Dense(sub_state_size*sub_state_size, activation='linear')
        self.out = layers.Dense(1, activation='tanh') # Single output which is Q-value

    def build(self, input_shape): # DK Adding this to stop Keras warning
        # Let the layers build automatically
        super().build(input_shape)

    def call(self, inputs):
        x = self.flatten(inputs)
        x = self.input_layer(x)
        x = self.d1(x)
        x = self.d2(x)
        x = self.d3(x)
        return self.out(x)

class ReplayBuffer:
    def __init__(self, capacity=5000):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)

    def store(self, s_sub, action, reward, s_sub_next, done):
        self.buffer.append((s_sub, action, reward, s_sub_next, done))

    def sample(self, batch_size=32):
        batch = random.sample(self.buffer, min(len(self.buffer), batch_size))
        s_subs, actions, rewards, s_subs_next, dones = [], [], [], [], []
        for s, a, r, s_next, d in batch:
            s_subs.append(s)
            actions.append(a)
            rewards.append(r)
            s_subs_next.append(s_next)
            dones.append(d)

        return (np.array(s_subs, dtype=np.float32),
                actions,
                np.array(rewards, dtype=np.float32),
                np.array(s_subs_next, dtype=np.float32),
                dones)

    def __len__(self):
        return len(self.buffer)
