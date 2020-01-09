import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import copy
from collections import namedtuple, deque
import random
import numpy as np

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Actor(nn.Module):
    def __init__(self,state_space_size,action_space_size,seed,fully_conected_units=256):
        super(Actor,self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fully_connected_1 = nn.Linear(state_space_size,fully_conected_units)
        self.fully_connected_2 = nn.Linear(fully_conected_units,action_space_size)
        self.params_reset()

    def forward(self,state):
        x = F.relu(self.fully_connected_1(state))
        return F.tanh(self.fully_connected_2(x))

    def params_reset(self):
        self.fully_connected_1.weight.data.uniform_(*limits(self.fully_connected_1))
        self.fully_connected_2.weight.data.uniform_(-3e-3,3e-3)

class Critic(nn.Module):
    def __init__(self,state_space_size,action_space_size,seed,hidden_1_units=256,hidden_2_units=256,hidden_3_units=128):
        super(Critic,self).__init__()
        self.seed = torch.manual_seed(seed)
        self.hidden_1 = nn.Linear(state_space_size,hidden_1_units)
        self.fully_connected_2 = nn.Linear(hidden_1_units+action_space_size,hidden_2_units)
        self.fully_connected_3 = nn.Linear(hidden_2_units,hidden_3_units)
        self.fully_connected_4 = nn.Linear(hidden_3_units,1)
        self.params_reset()

    def forward(self,state,action):
        t = F.leaky_relu(self.hidden_1(state))
        x = torch.cat((t,action),dim=1)
        x = F.leaky_relu(self.fully_connected_2(x))
        x = F.leaky_relu(self.fully_connected_3(x))
        return self.fully_connected_4(x)

    def params_reset(self):
        self.hidden_1.weight.data.uniform_(*limits(self.hidden_1)) 
        self.fully_connected_2.weight.data.uniform_(*limits(self.fully_connected_2)) 
        self.fully_connected_3.weight.data.uniform_(*limits(self.fully_connected_3))
        self.fully_connected_4.weight.data.uniform_(-3e-3,3e-3)  

class Memory:
    def __init__(self,action_space_size,buffer_size,batch_size,seed):
        self.action_space_size = action_space_size
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)
    
    def add_exp(self, state, action, reward, next_state, done):
        exp = self.experience(state, action, reward, next_state, done)
        self.memory.append(exp)
    
    def memory_sample(self):
        experiences = random.sample(self.memory, k=self.batch_size)
        states = torch.from_numpy(np.vstack([exp.state for exp in experiences if exp is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([exp.action for exp in experiences if exp is not None])).float().to(device)
        rewards = torch.from_numpy(np.vstack([exp.reward for exp in experiences if exp is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([exp.next_state for exp in experiences if exp is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([exp.done for exp in experiences if exp is not None]).astype(np.uint8)).float().to(device)
        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        return len(self.memory)


class Noise:
    def __init__(self,size,seed,mu=0.,theta=0.15,sigma=0.2):
        self.theta = theta
        self.sigma = sigma
        self.seed = random.seed(seed)
        self.mu = mu * np.ones(size)
        self.reset()

    def reset(self):
        self.state = copy.copy(self.mu)
    
    def noise_sample(self):
        x = self.state
        dx = self.theta*(self.mu-x)+self.sigma*np.array([random.random() for i in range(len(x))])
        self.state = x+dx
        return self.state

def limits(layer):
    inp = layer.weight.data.size()[0]
    limit = 1./np.sqrt(inp)
    return (-limit, limit)


class DDPG_Agent():
    def __init__(self,state_space_size,action_space_size,random_seed):
        self.state_space_size = state_space_size
        self.action_space_size = action_space_size
        self.seed = random.seed(random_seed)
        
        self.actor_local = Actor(state_space_size, action_space_size, random_seed).to(device)
        self.actor_target = Actor(state_space_size, action_space_size, random_seed).to(device)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=1e-4)

        self.critic_local = Critic(state_space_size, action_space_size, random_seed).to(device)
        self.critic_target = Critic(state_space_size, action_space_size, random_seed).to(device)
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=3e-4, weight_decay=0.0001)

        self.noise = Noise(action_space_size, random_seed)

        self.memory = Memory(action_space_size, int(1e6), 128, random_seed)

    def reset(self):
        self.noise.reset()

    def soft_update(self, local_model, target_model, tau):
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)

    def learn(self, experiences, gamma):
        states, actions, rewards, next_states, dones = experiences
        actions_next = self.actor_target(next_states)
        Q_targets_next = self.critic_target(next_states, actions_next)
        Q_targets = rewards+(gamma*Q_targets_next*(1-dones))
        Q_expected = self.critic_local(states,actions)
        critic_loss = F.mse_loss(Q_expected,Q_targets)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        actions_pred = self.actor_local(states)
        actor_loss = -self.critic_local(states, actions_pred).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        self.soft_update(self.critic_local, self.critic_target, 1e-3)
        self.soft_update(self.actor_local, self.actor_target, 1e-3)                     

    def current_action(self, state, add_noise=True):
        state = torch.from_numpy(state).float().to(device)
        self.actor_local.eval()
        with torch.no_grad():
            action = self.actor_local(state).cpu().data.numpy()
        self.actor_local.train()
        if add_noise:
            action += self.noise.noise_sample()
        return np.clip(action, -1, 1)

    def step(self, state, action, reward, next_state, done):
        self.memory.add_exp(state, action, reward, next_state, done)
        if len(self.memory) > 128:
            experiences = self.memory.memory_sample()
            self.learn(experiences, 0.99)

