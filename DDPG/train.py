import torch
import numpy as np
from collections import deque
import gym
import random

from walker import DDPG_Agent

env = gym.make('BipedalWalker-v2')
env.seed(10)
agent = DDPG_Agent(state_space_size=env.observation_space.shape[0],action_space_size=env.action_space.shape[0],random_seed=10)

def train_agent(episodes):
    max_timesteps = 700
    scores_deque = deque(maxlen=100)
    scores = []
    max_score = -np.Inf
    for episode in range(1,episodes+1):
        state = env.reset()
        agent.reset()
        episode_reward = 0
        for t in range(max_timesteps):
            action = agent.current_action(state)
            next_state, reward, done, _ = env.step(action)
            agent.step(state, action, reward, next_state, done)
            state = next_state
            episode_reward += reward
            if done:
                break 
        scores_deque.append(episode_reward)
        scores.append(episode_reward)
        print('\rEpisode {}\tAverage Reward: {:.2f}\t\tReward: {:.2f}'.format(episode, np.mean(scores_deque), episode_reward),end='')
        if episode % 100 == 0:
            torch.save(agent.actor_local.state_dict(), 'actor_checkpoint.pth')
            torch.save(agent.critic_local.state_dict(), 'critic_checkpoint.pth')
            print('\rEpisode {}\tAverage Reward: {:.2f}'.format(episode, np.mean(scores_deque)))   

        if np.mean(scores_deque)>0:
            break

train_agent(10000)
