#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 16 21:03:01 2020

@author: zixiangpei
"""

import functions as exampleCode
import random 
import numpy as np

class AgentInitializationClass():
    def __init__(self, env):
        self.env  = env
    
    def __call__(self, hAgent, dqnModel):
        return (exampleCode.AgentInitialization(hAgent, dqnModel, self.env))
    
class stepClass():
    def __init__(self, env):
        self.env = env
        
    def __call__(self, theta, h):
        return (exampleCode.step(theta,h, self.env))
    
class evalClass():
    def __init__(self, numSamples, env):
        self.numSamples = numSamples
        self.env = env
        
    def __call__(self, theta):
        return (exampleCode.eval(theta, self.numSamples, self.env))
    
class sampleExploitClass():
    def __call__(self, P, n):
        obj = random.sample(list(P), n)[0]
        return(obj)

class exploitClass():
    def __init__(self, sampleFunction, env):
        self.sampleFunction = sampleFunction
        self.env = env
    
    def __call__(self, h, theta, p, P):
        return (exampleCode.exploit(theta, h, p, P, self.sampleFunction, self.env))
    
class sampleExploreClass():
    def __call__(self,n):
        return([np.random.uniform(0.9, 1.1, 1)[0], random.sample([-4,-3,-2,-1,1,2,3,4], 1)[0]])

class exploreClass():
    def __init__(self, sampleFunction):
        self.sampleFunction = sampleFunction
        
    def __call__(self, hPrime, thetaPrime, P):
        return (exampleCode.explore(hPrime, thetaPrime, P, self.sampleFunction))
    
class endofTrainClass():
    def __init__(self, numIt):
        self.numIt = numIt
    
    def __call__(self, Population):
        return(Population[0]["t"] >= self.numIt)
    