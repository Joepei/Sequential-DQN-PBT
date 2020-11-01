#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 23 16:42:26 2020
@author: zixiangpei
"""
import gym
import numpy as np
import unittest
from ddt import ddt, data, unpack
#import DQNCallable as callableCode
import functions as targetCode
import q_net as qnet
import q_agent as qagent
import tensorflow as tf

@ddt
class TestToyExample(unittest.TestCase):
    def setUp(self):
        env = gym.make('MountainCar-v0')
        dqnModel1 = qnet.build_dqn_model(0.01) #layer number, node in each layer
        dqnModel2 = qnet.build_dqn_model(0.01)
        for i in range(5):
            dqnModel1.fit(tf.convert_to_tensor([[1,1]]),tf.convert_to_tensor([[2,2,2]])) #some random params to make sure the models don't have arbitrary weights 
        for i in range(5):
            dqnModel2.fit(tf.convert_to_tensor([[3,3]]),tf.convert_to_tensor([[4,4,4]])) 
            
        agent1 = targetCode.AgentInitialization([0.99,1], dqnModel1, env)
        agent2 = targetCode.AgentInitialization([0.98,12], dqnModel2, env)
        self.agent1 = agent1
        self.agent2 = agent2
        self.exploitSample1 = lambda x,y: {"theta" : {"agent": agent1, "h":[0.99,1], "id":1},"h": [0.99,1], "p": {"p":12,"id":1}, "t": 4}
        self.exploitSample2 = lambda x,y: {"theta" : {"agent": agent2, "h":[0.98,12], "id":2},"h": [0.98,12], "p": {"p":15,"id":2}, "t": 8}
        self.exploreSamplePositive = lambda n: (0.99, 1)
        self.exploreSampleNegative = lambda n: (0.98, -1)
        self.P = [{"theta" : {"agent": agent1, "h":[0.99,1], "id":1},"h": [0.99,1], "p": {"p":12,"id":1}, "t": 4}, {"theta" : {"agent": agent2, "h":[0.98,12], "id":2},"h": [0.98,12], "p": {"p":15,"id":2}, "t": 8}]
        
    
    #given an agent I can get its learning rate and batch size
    #restorehNeural
    #restorehNonNeural
    
    @data(([0.99,1]))
    def testAgentInitilizaitonWorker1(self, expectedResult):
        print(1)
        self.assertEqual(self.agent1.epsDecRate, expectedResult[0])
        self.assertEqual(self.agent1.batchSize, expectedResult[1])
        
    @data(([0.98,12]))
    def testAgentInitilizaitonWorker2(self, expectedResult):
        print(2)
        self.assertEqual(self.agent2.epsDecRate, expectedResult[0])
        self.assertEqual(self.agent2.batchSize, expectedResult[1])
    
    
    @data(([[0.99,1],  {"p":12,"id":1}, 4], [[2,2]]))
    @unpack
    def testInitializationAgent1(self, expectedResult, sampleInput):
        print(3)
        self.assertListEqual(self.P[0]["h"], expectedResult[0])
        self.assertDictEqual(self.P[0]["p"], expectedResult[1])
        self.assertEqual(self.P[0]["t"], expectedResult[2])
        resultAssigned = self.P[0]["theta"]["agent"].qNetwork().predict(tf.convert_to_tensor(sampleInput))
        resultOld  = self.agent1.qNetwork().predict(tf.convert_to_tensor(sampleInput))
        np.testing.assert_array_almost_equal(resultAssigned[0], resultOld[0])
        
    @data(([[0.98,12],  {"p":15,"id":2}, 8], [[2,2]]))
    @unpack
    def testInitializationAgent2(self, expectedResult, sampleInput):
        print(4)
        self.assertListEqual(self.P[1]["h"], expectedResult[0])
        self.assertDictEqual(self.P[1]["p"], expectedResult[1])
        self.assertEqual(self.P[1]["t"], expectedResult[2])
        resultAssigned = self.P[1]["theta"]["agent"].qNetwork().predict(tf.convert_to_tensor(sampleInput))
        resultOld  = self.agent2.qNetwork().predict(tf.convert_to_tensor(sampleInput))
        np.testing.assert_array_almost_equal(resultAssigned[0], resultOld[0])
    
    
    env = gym.make('MountainCar-v0')
    dqnModel1 = qnet.build_dqn_model(0.01)
    agent1 = qagent.Agent(gamma = 0.99, numActions= env.action_space.n, dqnModel = dqnModel1, epsilon=1.0, batchSize = 1, epsDecRate = 0.99, epsMin = 0.1)
    
    @data(({"agent": agent1, "h": [0.99, 1], "id": 1}, [0.99,1], env, [[-4.79472897e-01, 8.94804181e-03]], [[-2.6321998, -1.9456255, -2.5451536]]))
    @unpack
    def teststep(self, theta, h, env, states, qTarget):
        print(5)
        modelBeforeTrain = tf.keras.models.clone_model(theta["agent"].qNetwork())
        state = env.reset()
        done = False 
        action = theta["agent"].getAction(state)
        nextState, reward, done, info = env.step(action)
        states, qTarget = theta["agent"].train(state, action, reward, nextState, done)
        state = nextState
        modelAfterTrain = tf.keras.models.clone_model(theta["agent"].qNetwork())
        modelAfterTrain.set_weights(theta["agent"].qNetwork().get_weights())
        resultBefore = modelBeforeTrain.predict(tf.convert_to_tensor(states))
        resultAfter = modelAfterTrain.predict(tf.convert_to_tensor(states))
        lossBefore = tf.keras.losses.MSE(tf.convert_to_tensor(qTarget), resultBefore)
        lossAfter = tf.keras.losses.MSE(tf.convert_to_tensor(qTarget), resultAfter)
        self.assertTrue(lossBefore.numpy()[0] > lossAfter.numpy()[0])
    
        
    """
    env = gym.make("MountainCar-v0")
    dqnModelLow = qnet.build_dqn_model(0.01)
    dqnModelHigh = qnet.build_dqn_model(0.01)#make this a clone of modelLow
    agentLow = qagent.Agent(gamma = 0.99, numActions = env.action_space.n, dqnModel = dqnModelLow, epsilon = 1.0, batchSize = 64, epsDecRate = 0.99, epsMin = 0.1)
    agentHigh = qagent.Agent(gamma = 0.99, numActions = env.action_space.n, dqnModel = dqnModelHigh, epsilon = 1.0, batchSize = 64, epsDecRate = 0.99, epsMin = 0.1)
    @data(({"agent": agentHigh, "h": [0.99, 64], "id": 1},{"agent":agentLow, "h":[0.99,64], "id":2}, [0.99, 64], 20, 400, env))
    @unpack
    def testeval(self, thetaHigh,thetaLow, h, numSamples, numIter, env):
        for i in range(numIter):
            thetaHigh = targetCode.step(thetaHigh, h, env)
            print(i)
        evalHigh = targetCode.eval(thetaHigh, numSamples, env)
        evalLow = targetCode.eval(thetaLow, numSamples, env)
        print(evalHigh["p"])
        print(evalLow["p"])
        self.assertGreater(evalHigh["p"], evalLow["p"])
    """
    
    @data(({"p":12,"id":1}, 204, True), ({"p":15,"id":2}, 89,  False))
    @unpack
    def testready(self, p, t, expectedResult):
        print(6)
        state = targetCode.ready(p, t, self.P)
        self.assertEqual(state, expectedResult)
        
    @data(([0.99,1], {"agent": agent1, "h": [0.99, 1], "id": 1}, {"p":12,"id":1}, env, [[3,3]], [[4,4,4]]))
    @unpack
    def testExploitLose(self, h, theta, p, env, sampleInput, sampleOutput):
        print(7)
        hPrime, thetaPrime = targetCode.exploit(h, theta, p, self.P, self.exploitSample2, env)
        self.assertListEqual(hPrime, self.P[1]["h"])
        resultthetaPrime = thetaPrime["agent"].qNetwork().predict(tf.convert_to_tensor(sampleInput))
        lossthetaPrime = tf.keras.losses.MSE(tf.convert_to_tensor(sampleOutput), resultthetaPrime)
        resultOpponent = self.P[1]["theta"]["agent"].qNetwork().predict(tf.convert_to_tensor(sampleInput))
        lossOpponent = tf.keras.losses.MSE(tf.convert_to_tensor(sampleOutput), resultOpponent)
        self.assertEqual(lossthetaPrime.numpy()[0], lossOpponent.numpy()[0])
        self.assertEqual(thetaPrime["id"], theta["id"])
        
    dqnModel2 = qnet.build_dqn_model(0.01)
    agent2 = qagent.Agent(gamma = 0.99, numActions= env.action_space.n, dqnModel = dqnModel2, epsilon=1.0, batchSize = 12, epsDecRate = 0.98, epsMin = 0.1)
    @data(([0.98,12], {"agent": agent2, "h":[0.98,12], "id": 2}, {"p":15, "id":2}, env, [[3,3]], [[4,4,4]]))
    @unpack
    def testExploitWin(self, h, theta, p, env, sampleInput, sampleOutput):
        print(8)
        hPrime, thetaPrime = targetCode.exploit(h, theta, p, self.P, self.exploitSample1, env)
        self.assertListEqual(hPrime, self.P[1]["h"])
        ##warning appears somewhere here
        resultthetaPrime = thetaPrime["agent"].qNetwork().predict(tf.convert_to_tensor(sampleInput))
        lossthetaPrime = tf.keras.losses.MSE(sampleOutput, resultthetaPrime)
        resultSelf = theta["agent"].qNetwork().predict(tf.convert_to_tensor(sampleInput))
        lossOpponent = tf.keras.losses.MSE(sampleOutput, resultSelf)
        self.assertEqual(lossthetaPrime.numpy()[0], lossOpponent.numpy()[0])
        self.assertEqual(thetaPrime["id"], theta["id"])
        
    @data(([0.99,1], {"agent": agent1, "h": [0.99, 1],"id": 1}, [0.98109,2]))
    @unpack
    def testExploreSamplePositive1(self, hPrime, thetaPrime, expectedResult):
        print(9)
        newh, newtheta = targetCode.explore(hPrime, thetaPrime, self.P, self.exploreSamplePositive)
        self.assertListEqual(newh, expectedResult)
        self.assertListEqual(newtheta["h"], expectedResult)
        
    @data(([0.98,12], {"agent": agent1, "h": [0.98, 12], "id": 2}, [0.96138, 11]))
    @unpack
    def testExploreSampleNegative2(self, hPrime, thetaPrime, expectedResult):
        print(10)
        newh, newtheta = targetCode.explore(hPrime, thetaPrime, self.P, self.exploreSampleNegative)
        self.assertListEqual(newh, expectedResult)
        self.assertListEqual(newtheta["h"], expectedResult)
        
    @data(({"agent": agent2, "h":[0.98,12], "id": 1}, [0.98,12], {"p":15, "id":1}, 12, [[3,3]]))
    @unpack
    def testUpdate(self, theta, h, p, t, sampleInput):
        print(11)
        P, newt = targetCode.update(self.P, theta, h, p, t)
        self.assertListEqual(P[0]["h"], h)
        self.assertDictEqual(P[0]["p"], p)
        self.assertEqual(P[0]["t"], t+1)
        self.assertEqual(newt, t+1)
        resultUpdatedNet = P[0]["theta"]["agent"].qNetwork().predict(tf.convert_to_tensor(sampleInput))
        resultTargetNet = theta["agent"].qNetwork().predict(tf.convert_to_tensor(sampleInput))
        np.testing.assert_array_almost_equal(resultUpdatedNet[0],resultTargetNet[0])

        
if __name__ == '__main__':
    unittest.main(verbosity=2)