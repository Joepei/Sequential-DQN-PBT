import numpy as np
import tensorflow as tf
import random
"""
import gym
from tensorflow import keras
from collections import deque
from q_net import build_dqn_model
"""
from replay_buffer import ReplayBuffer
import keras_pickle_wrapper


class Agent():
    def __init__(self, gamma, numActions, dqnModel, epsilon =1.0 , batchSize = 64, epsDecRate = 0.99, epsMin=0.01,
                maxSize=100000):
        self.gamma = gamma
        self.numActions = numActions
        self.epsilon = epsilon
        self.batchSize = batchSize
        self.epsDecRate = epsDecRate
        self.epsMin = epsMin
        self.memory = ReplayBuffer(maxSize)
        self.qNetwork = keras_pickle_wrapper.KerasPickleWrapper(dqnModel)
        
    def getAction(self, state):
        stateInput = tf.convert_to_tensor(np.array([state]))
        qStates = self.qNetwork().predict(stateInput)
        actionGreedy = np.argmax(qStates)
        actionRandom = np.random.randint(self.numActions)
        if random.random() > self.epsilon: #epsilon-greedy algorithm
            action = actionGreedy
        else:
            action = actionRandom
        return action

    def train(self, state, action, reward, nextState, done):
        self.memory.store(state, action, reward, nextState, done)
        
        if self.memory.mem_cntr < self.batchSize: #Only train when there is enough sample for a batch
            return([[]],[[]])
            
        states, actions, rewards, nextStates, dones = self.memory.sample(self.batchSize)
        qValue = self.qNetwork().predict(tf.convert_to_tensor(states))
        qNextValue = self.qNetwork().predict(tf.convert_to_tensor(nextStates))
        qTarget = np.copy(qValue)
        
        for i in range(states.shape[0]):
            qTarget[i, actions[i]] = rewards[i] + self.gamma * np.max(qNextValue[i])*(1-dones[i])
        
        
        self.qNetwork().train_on_batch(tf.convert_to_tensor(states), tf.convert_to_tensor(qTarget))
        return(states, qTarget)
