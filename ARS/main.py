from normalizer import Normalizer
from policy import Policy
import gym
import numpy as np

epochs = 1000
lr = 0.02
num_dirs = 16
total_episodes = 2000
num_dirs_best = 16
noise = 0.03

def exploration(env, normalizer, policy, direction=None, delta=None):
    state = env.reset()
    done = False
    episode = 0.
    sigma_rewards = 0
    while not done and episode < total_episodes:
        normalizer.observe(state)
        state = normalizer.normalize(state)
        action = policy.evaluate(state, delta, direction)
        state, reward, done, _ = env.step(action)
        reward = max(min(reward, 1), -1)
        sigma_rewards += reward
        episode += 1
    return sigma_rewards

def train_ars_agent(env,policy,normalizer,epochs,lr,num_dirs,total_episodes,num_dirs_best,noise = 0.03):
    for epoch in range(epochs):
        deltas = policy.sample_deltas()
        positive_rewards = [0] * num_dirs
        negative_rewards = [0] * num_dirs

        for i in range(num_dirs):
            positive_rewards[i] = exploration(env,normalizer,policy,direction="positive",delta=deltas[i])
        for i in range(num_dirs):
            negative_rewards[i] = exploration(env,normalizer,policy,direction="negative",delta=deltas[i])

        final_rewards = np.array(positive_rewards+negative_rewards)
        sigma_r = final_rewards.std()

        scores = {score:max(positive_reward, negative_reward) for score, (positive_reward, negative_reward) in enumerate(zip(positive_rewards, negative_rewards))}

        order = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)[:num_dirs_best]
        rollouts = [(positive_rewards[k], negative_rewards[k], deltas[k]) for k in order]
        policy.update(rollouts, sigma_r)
        reward_eval = exploration(env,normalizer,policy)

        print('Epoch:', epoch, '   -------------X--------------   Reward:', reward_eval)


if __name__ == '__main__':
    np.random.seed(1)
    env = gym.make('BipedalWalker-v3')
    env = gym.wrappers.Monitor(env,'Runs',force=True)   
    policy = Policy(env.observation_space.shape[0],env.action_space.shape[0],lr,num_dirs,num_dirs_best,noise)
    normalizer = Normalizer(env.observation_space.shape[0])
    train_ars_agent(env,policy,normalizer,epochs,lr,num_dirs,total_episodes,num_dirs_best,noise)