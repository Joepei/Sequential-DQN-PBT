from collections import deque
import numpy as np
import random

class ReplayBuffer():
    def __init__(self, max_size):
        self.buffer = deque(maxlen=max_size)
        self.mem_cntr = 0
        
    def store(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
        self.mem_cntr += 1
    
    def sample(self, batch_size):
        sample_size = min(batch_size, self.mem_cntr)
        samples = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = [], [], [], [], []
        for sample in samples:
            states.append(sample[0])
            actions.append(sample[1])
            rewards.append(sample[2])
            next_states.append(sample[3])
            dones.append(sample[4])
        return np.array(states), actions, rewards, np.array(next_states), dones
