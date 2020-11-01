#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 16 22:27:10 2020

@author: zixiangpei
"""
import os
import time
import psutil

process = psutil.Process(os.getpid())
        
import time
class PBTClass:
    def __init__(self, step, eval, ready, exploit, explore, update, endofTrain, Population):
        self.step = step
        self.eval = eval
        self.ready = ready
        self.exploit = exploit
        self.explore = explore
        self.update = update
        self.P = Population
        self.endofTrain = endofTrain
        self.k = 0
        self.exploitFlag = True
        self.exploreFlag = True

    def train(self):
        Q0 = []
        Q1 = []
        while not self.endofTrain(self.P):
            for value in self.P:
                theta, h, p, t = (value["theta"], value["h"], value["p"], value["t"])
                print(t)
                theta = self.step(theta, h)
                p = self.eval(theta)
                if(t % 5 == 0):
                    if(p["id"] == 1):
                        Q0.append(p["p"])
                    else:
                        Q1.append(p["p"])
                """
                if self.ready(p,t,self.P):
                    hPrime, thetaPrime = self.exploit(h, theta, p, self.P) ###Problematic
                    if theta != thetaPrime:
                        h, theta = self.explore(hPrime, thetaPrime, self.P)
                        p = self.eval(theta)
                """
                print(process.memory_info().rss)
                self.P, t = self.update(self.P, theta, h, p, t)
        return(Q0, Q1)