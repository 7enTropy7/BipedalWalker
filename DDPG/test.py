import gym
import random
import torch
import numpy as np
from collections import deque
import matplotlib.pyplot as plt

from walker import DDPG_Agent

env = gym.make('BipedalWalker-v2')
env.seed(10)
agent = DDPG_Agent(state_space_size=env.observation_space.shape[0], action_space_size=env.action_space.shape[0], random_seed=10)

agent.actor_local.load_state_dict(torch.load('actor_checkpoint.pth'))
agent.critic_local.load_state_dict(torch.load('critic_checkpoint.pth'))

while True:
    state = env.reset()
    agent.reset()   

    while True:
        action = agent.current_action(state)
        env.render()
        next_state, reward, done, _ = env.step(action)
        state = next_state
        if done:
            break
            
    env.close()