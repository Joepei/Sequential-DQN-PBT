import numpy as np
import tensorflow as tf
import q_agent as agentClass
#theta: {"agent": agent, "h": [h0, h1], "id": id, t: }
#h: [h0, h1]
#p: {"p": p, "id": id}
#t: t

def AgentInitialization(hAgent, dqnModel, env):
    agent = agentClass.Agent(gamma = 0.99, numActions= env.action_space.n, dqnModel = dqnModel, epsilon=1.0, batchSize = hAgent[1], epsDecRate = hAgent[0], epsMin = 0.1)
    return (agent)
    

def step(theta, h, env):
    state = env.reset()
    done = False 
    totalReward = 0 
    while not done:
        action = theta["agent"].getAction(state)
        nextState, reward, done, info = env.step(action)
        totalReward += reward
        states, qTarget = theta["agent"].train(state, action, reward, nextState, done)
        state = nextState
    #get final model 
    if theta["agent"].epsilon > theta["agent"].epsMin:
            theta["agent"].epsilon = theta["agent"].epsilon * theta["agent"].epsDecRate
    return(theta)
        
def eval(theta, numSamples, env):
    #sample 100 trajectory 
    #always greedy action 
    #calculate average reward
    totalReward = 0 
    for i in range(numSamples):
        state = env.reset()
        done = False
        totalRewardPerEpi = 0
        while not done: 
            stateInput = np.array([state])
            qStates = theta["agent"].qNetwork().predict(stateInput)
            actionGreedy = np.argmax(qStates) #To make the point that we are always doing action greedy
            nextState, reward, done, info = env.step(actionGreedy)
            totalRewardPerEpi += reward
            state = nextState
        totalReward = totalReward + totalRewardPerEpi
    avgReward = totalReward/numSamples             
    p = {}
    p["p"] = avgReward
    p["id"] = theta["id"]
    return(p)


def ready(p, t, P):
    return(t%5==4 and t > 200)

def exploit(h, theta, p, P, sampleFunction, env):
    opponent = sampleFunction(P, 1)
    while opponent["p"]["id"] == p["id"]: #to ensure that opponent is different from the worker itself
        opponent = sampleFunction(P,1)
    if opponent["p"]["p"] >= p["p"]:
        opponentQnet = tf.keras.models.clone_model(opponent["theta"]["agent"].qNetwork())
        opponentQnet.set_weights(opponent["theta"]["agent"].qNetwork().get_weights())
        thetaPrime = {}
        thetaPrime["h"] = opponent["h"][:]
        thetaPrime["p"] = opponent["p"]["p"]
        thetaPrime["id"] = theta["id"]
        thetaPrime["agent"] = agentClass.Agent(gamma = theta["agent"].gamma, numActions= env.action_space.n, dqnModel = opponentQnet, epsilon=theta["agent"].epsilon, batchSize = thetaPrime["h"][1], epsDecRate = thetaPrime["h"][0], epsMin = theta["agent"].epsMin)
        h = opponent["h"][:]
        return (h, thetaPrime)
    else:
        return (h, theta)   
    
def explore(hPrime, thetaPrime, P, sampleFunction):
    h0Perturb, h1Perturb = sampleFunction(len(hPrime))
    newh = hPrime[:]
    newh[0] = (hPrime[0]+0.001) * h0Perturb
    newh[1] = hPrime[1] + h1Perturb
    thetaPrime["h"] = newh[:]
    return(newh, thetaPrime)

def update(P, theta, h, p, t):
    for counter, item in enumerate(P):
        if(item["p"]["id"] == p["id"]):
            P[counter]["h"] = h[:]
            P[counter]["p"] = p.copy()
            P[counter]["t"] = t+1
            P[counter]["theta"] = theta.copy()
    return(P, t+1)
