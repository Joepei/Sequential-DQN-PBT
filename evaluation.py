import numpy as np
import tensorflow as tf
import gym
from tensorflow import keras
from collections import deque
import random
from replay_buffer import ReplayBuffer
from q_net import build_dqn_model
from q_agent import Agent
import pandas as pd
from collections import OrderedDict
import random
import pylab as plt
import pickle


x = 0

def evaluateDQN(df):
    global x
    eps_min = df.index.get_level_values('eps_min')[0]
    eps_dec_rate =  df.index.get_level_values('eps_dec_rate')[0]
    gamma =  df.index.get_level_values('gamma')[0]
    env = gym.make('MountainCar-v0')
    num_episodes = 3
    lr = 0.001
    dqn_model = build_dqn_model(0.001)
    agent = Agent(gamma = gamma, num_actions=env.action_space.n, dqn_model = dqn_model, epsilon=1.0, eps_dec_rate = eps_dec_rate, eps_min = eps_min)
    episodes = []
    total_rewards = []
    avg_reward = -200
    series = pd.Series()
    for i in range(num_episodes):
        episodes.append(i)
        done = False
        total_reward = 0 
        state = env.reset()
        while not done:
            action = agent.get_action(state)
            next_state, reward, done, info = env.step(action)
            total_reward += reward
            agent.train(state, action, reward, next_state, done)
            #env.render()
            state = next_state
        if agent.epsilon > agent.eps_min:
            agent.epsilon = agent.epsilon * agent.eps_dec_rate
        series.at[i] = total_reward
        total_rewards.append(total_reward)
        avg_reward = np.mean(total_rewards[-100:])
        print("Episode: {}, Total_reward: {:.2f}, Avg_reward_last_100_games: {:.2f}".format(i, total_reward, avg_reward))
    # return(pd.Series({'total_reward' : total_rewards, 'episodes' : episodes}))
    x += 1
    return(series)



def main():
    independentVariables = OrderedDict()
    independentVariables['eps_min'] = [0.01,0.05,0.1]
    independentVariables['eps_dec_rate'] = [0.97, 0.95, 0.9]
    independentVariables['gamma'] = [0.9, 0.95, 0.99]

    fig = plt.figure(figsize = (20,15))
    levelNames = list(independentVariables.keys())
    levelValues = list(independentVariables.values())
    levelIndex = pd.MultiIndex.from_product(levelValues, names=levelNames)
    toSplitFrame = pd.DataFrame(index = levelIndex)
    #first_half = toSplitFrame.iloc[pd.np.r_[0,21:27]]
    modelResultDF = toSplitFrame.groupby(levelNames).apply(evaluateDQN)
    #modelResultDF.to_csv('evaluation.csv', mode='a', header=False)
    #modelResultDF.to_csv('evaluation.csv')

    plotRowNum = len(independentVariables['eps_min'])
    plotColNum = len(independentVariables['eps_dec_rate'])
    plotCounter = 1

    for key, plotDf in modelResultDF.groupby('eps_min'):
        plotDf.index = plotDf.index.droplevel('eps_min')
        for keyRow, innerSubDF in plotDf.groupby('eps_dec_rate'):
            ax = fig.add_subplot(plotRowNum, plotColNum, plotCounter)
            ax.set_xticks([100,200,300])
            innerSubDF = innerSubDF.droplevel('eps_dec_rate').T
            innerSubDF.plot.line(ax = ax)
            ax.set_title('eps_min =' + str(key) + ', eps_dec_rate = ' + str(keyRow) + ' ')
            ax.set_xlabel('Number of Episodes')
            ax.set_ylabel('Rewards per Episode')
            ax.legend(title = 'gamma', loc='upper left')
            plotCounter += 1

    plt.title('Evaluation on [eps_min, eps_dec_rate, gamma] and How They Affect Total Reward')
    fig.savefig("evaluation.png")
    plt.show()


if __name__ == '__main__':
    main()


# figure = plt.figure(figsize = (10, 5))
#for keys, plotDf in modelResultDF.groupby(')







