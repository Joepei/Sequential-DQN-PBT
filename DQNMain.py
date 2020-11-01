import sys
sys.path.append('../DQN_PBT')
import numpy as np
import functions as exampleCode
import PBTClass as algoCode
import random
import pylab as plt
import multiprocessing
import pickle
from DQN_Callable import AgentInitializationClass
from DQN_Callable import stepClass
from DQN_Callable import evalClass
from DQN_Callable import sampleExploreClass
from DQN_Callable import exploitClass
from DQN_Callable import exploreClass
from DQN_Callable import endofTrainClass
from DQN_Callable import sampleExploitClass
from queue import Queue
import time
import gym
import functions as exampleCode
import q_net as qnet
import q_agent as agentClass
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}
import warnings
warnings.filterwarnings('ignore')
from silence_tensorflow import silence_tensorflow
silence_tensorflow()
"""

def plotQ(Q0, Q1):
    """
    Q0 = QLists[0]
    Q1 = QLists[1]
    """
    index = list(range(0,len(Q0)))
    fig = plt.figure()
    plt.plot(index, Q0)
    plt.plot(index, Q1, 'r-')
    plt.ylim(-201, -100)
    plt.xlabel("steps")
    plt.ylabel("Q(theta)")
    plt.title("PBT")
    plt.xticks(range(0,len(Q0)))
    plt.show()
    fig.savefig("PBT_Q.png")

def dump_queue(queue):
    result = []
    while not queue.empty():
        item = queue.get()
        result.append(item)
    return(result)

def process(trainer, Population):
        pop_length = len(Population)
        """
        QQueues = [] #create Queues here
        #QQueues  = multiprocessing.Queue()
        thetaQueues = []
        ps = []
        workers = []
        """
        #create a function
        """
        for i in range(pop_length): #Use enumerate
            QQueues.append(multiprocessing.Queue())
            thetaQueues.append(multiprocessing.Queue())
            initial = Population[i]
            workers.append((initial, QQueues[i], thetaQueues[i], bestthetaQueue))
            ps.append(multiprocessing.Process(target = trainer, args = (workers[i],)))
        """
        def prepare(initial):
            QQueue = multiprocessing.Queue()
            worker = (initial, QQueue)
            process = multiprocessing.Process(target = trainer, args = (worker,))
            return(QQueue, process)
        
        QueueList = [prepare(initial) for initial in Population]
        QQueues = [QQueue for (QQueue, process) in QueueList]
        #thetaQueues = [thetaQueue for (QQueue, thetaQueue, process) in QueueList]
        ps = [process for (QQueue, process) in QueueList]
        
        """
        for initial in Population:
            QQueue = multiprocessing.Queue()
            thetaQueue = multiprocessing.Queue()
            worker = (initial, QQueue, thetaQueue, bestthetaQueue)
            process = multiprocessing.Process(target = trainer, args = (worker,))
            QQueues.append(QQueue)
            thetaQueues.append(thetaQueue)
            workers.append(worker)
            ps.append(process)
        """
            
        print("ready to go")
        startTime = time.time()
        for process in ps:
            import tensorflow as tf
            from tensorflow.compat.v1.keras.backend import set_session
            set_session(tf.compat.v1.Session())
            process.start()
        
        for process in ps:
            process.join()
        endTime = time.time()
        print(endTime - startTime)
        """
        #get all the lists here
        for i in range(pop_length):
            ps[i].start()
            time.sleep(0.015)
        for i in range(pop_length):
            ps[i].join()
        """

        QLists = []
        for i in range(pop_length):
            QLists.append(dump_queue(QQueues[i]))
            #thetaLists.append(dump_queue(thetaQueues[i]))
        #bestthetaList = dump_queue(bestthetaQueue)

        plotQ(QLists)
        #return(bestthetaList)


def main():
    numIt = 20
    numSamples = 2
    env = gym.make('MountainCar-v0')
    dqnModel1 = qnet.build_dqn_model(0.005)
    dqnModel2 = qnet.build_dqn_model(0.005)
    AgentInitialization = AgentInitializationClass(env)
    agent1 = AgentInitialization([0.99, 64], dqnModel1)
    agent2 = AgentInitialization([0.95, 16], dqnModel2)
    #agent1 = agentClass.Agent(gamma = 0.99, numActions= env.action_space.n, dqnModel = dqnModel1, epsilon=1.0, batchSize = 64, epsDecRate = 0.99, epsMin = 0.1)
    #agent2 = agentClass.Agent(gamma = 0.99, numActions= env.action_space.n, dqnModel = dqnModel2, epsilon=1.0, batchSize = 16, epsDecRate = 0.95, epsMin = 0.1)
    step = stepClass(env)
    eval = evalClass(numSamples, env)
    ready = exampleCode.ready
    sampleExploit = sampleExploitClass()
    exploit = exploitClass(sampleExploit, env)
    sampleExplore = sampleExploreClass()
    explore = exploreClass(sampleExplore)
    update = exampleCode.update
    endofTrain = endofTrainClass(numIt)
        
    Population = [{"theta" : {"agent": agent1, "h":[0.99,64], "id":1},\
                                                 "h": [0.99,64], "p": {"p":0,"id":1}, "t": 0}, \
                                   {"theta" : {"agent": agent2, "h":[0.95,16], "id":2},\
                                                 "h": [0.95,16], "p": {"p":0,"id":2}, "t": 0}]
    PBTObject = algoCode.PBTClass(step, eval, ready, exploit, explore, update,  endofTrain, Population)
    Q0, Q1= PBTObject.train()
    plotQ(Q0, Q1)
    
if __name__ == '__main__':
    main()


