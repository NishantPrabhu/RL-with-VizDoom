
""" 
Agents 
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import networks 
import random


class DQNAgent:

    def __init__(self, config, action_size, device):
        self.config = config
        self.action_size = action_size
        self.gamma = self.config['gamma']
        self.frame_repeat = self.config['frame_repeat']
        self.batch_size = self.config['batch_size']
        self.epsilon = self.config['epsilon']
        self.eps_decay_rate = self.config['eps_decay_rate']
        self.eps_min = self.config['eps_min']
        self.device = device

        self.replay_memory = networks.ReplayMemory(
            memory_size = self.config['replay_memory_size'], 
            batch_size = self.config['batch_size'], 
            resolution = self.config['resolution'], 
            device = self.device)

        # Model, optimizer, loss functions
        self.model = networks.DQN(action_size).to(self.device) 
        self.optim = optim.SGD(self.model.parameters(), lr=self.config['optim_lr'], momentum=0.9)
        self.criterion = nn.MSELoss()

    def get_action(self, state):
        if random.uniform(0, 1) < self.epsilon:
            return random.choice(range(self.action_size))
        else:
            state = np.expand_dims(state, axis=0)                                                       # (1, 3, W, H)    
            state = torch.from_numpy(state).float().to(self.device)                                     # (1, 3, W, H)
            with torch.no_grad():
                action = self.model(state).argmax(dim=-1)                                                   # (1, 1)
            return action.item() 

    def learn_from_memory(self):
        state, action, reward, new_state, is_done = self.replay_memory.get_transactions()
        output = self.model(state).gather(1, action.view(-1, 1)).squeeze(-1)                            # (batch_size,)
        with torch.no_grad():
            target = reward + self.gamma * (1-is_done) * self.model(new_state).max(dim=-1)[0]           # (batch_size,)
        
        # Update model
        self.optim.zero_grad()
        loss = self.criterion(target, output)
        loss.backward()
        self.optim.step()

        # Decay epsilon
        if self.epsilon > self.eps_min:
            self.epsilon *= self.eps_decay_rate
        else:
            self.epsilon = self.eps_min

        return {'Loss': loss.item()}


class DoubleDQNAgent:

    def __init__(self, config, action_size, device):
        self.config = config
        self.action_size = action_size
        self.gamma = self.config['gamma']
        self.frame_repeat = self.config['frame_repeat']
        self.batch_size = self.config['batch_size']
        self.epsilon = self.config['epsilon']
        self.eps_decay_rate = self.config['eps_decay_rate']
        self.eps_min = self.config['eps_min']
        self.device = device

        self.replay_memory = networks.ReplayMemory(
            memory_size = self.config['replay_memory_size'], 
            batch_size = self.config['batch_size'], 
            resolution = self.config['resolution'], 
            device = self.device)

        # Model, optimizer, loss functions
        self.model = networks.DQN(action_size).to(self.device)
        self.target_model = networks.DQN(action_size).to(self.device).eval()
        self.optim = optim.SGD(self.model.parameters(), lr=self.config['optim_lr'], momentum=0.9)
        self.criterion = nn.MSELoss()

    def get_action(self, state):
        if random.uniform(0, 1) < self.epsilon:
            return random.choice(range(self.action_size))
        else:
            state = np.expand_dims(state, axis=0)                                                           # (1, 3, W, H)
            state = torch.from_numpy(state).float().to(self.device)                                         # (1, 3, W, H)
            with torch.no_grad():
                action = self.model(state).argmax(dim=-1)                                                       # (1, 1)
            return action.item() 

    def learn_from_memory(self):
        state, action, reward, new_state, is_done = self.replay_memory.get_transactions()
        output = self.model(state).gather(1, action.view(-1, 1)).squeeze(-1)                                # (batch_size,)
        with torch.no_grad():
            best_actions = self.model(new_state).argmax(dim=-1)
            new_state_values = self.target_model(new_state).gather(1, best_actions.view(-1, 1)).squeeze(-1)
            target = reward + self.gamma * (1-is_done) * new_state_values                                   # (batch_size,)
        
        # Update model
        self.optim.zero_grad()
        loss = self.criterion(output, target)
        loss.backward()
        self.optim.step()

        # Decay epsilon
        if self.epsilon > self.eps_min:
            self.epsilon *= self.eps_decay_rate
        else:
            self.epsilon = self.eps_min

        return {'Loss': loss.item()}


class DuelDQNAgent:

    def __init__(self, config, action_size, device):
        self.config = config
        self.action_size = action_size
        self.gamma = self.config['gamma']
        self.frame_repeat = self.config['frame_repeat']
        self.batch_size = self.config['batch_size']
        self.epsilon = self.config['epsilon']
        self.eps_decay_rate = self.config['eps_decay_rate']
        self.eps_min = self.config['eps_min']
        self.device = device

        self.replay_memory = networks.ReplayMemory(
            memory_size = self.config['replay_memory_size'], 
            batch_size = self.config['batch_size'], 
            resolution = self.config['resolution'], 
            device = self.device)

        # Model, optimizer, loss functions
        self.model = networks.DQN(action_size).to(self.device)
        self.target_model = networks.DQN(action_size).to(self.device).eval()
        self.optim = optim.SGD(self.model.parameters(), lr=self.config['optim_lr'], momentum=0.9)
        self.criterion = nn.MSELoss()

    def get_action(self, state):
        if random.uniform(0, 1) < self.epsilon:
            return random.choice(range(self.action_size))
        else:
            state = np.expand_dims(state, axis=0)                                                           # (1, 3, W, H)
            state = torch.from_numpy(state).float().to(self.device)                                         # (1, 3, W, H)
            with torch.no_grad():
                action = self.model(state).argmax(dim=-1)                                                       # (1, 1)
            return action.item() 

    def learn_from_memory(self):
        state, action, reward, new_state, is_done = self.replay_memory.get_transactions()
        output = self.model(state).gather(1, action.view(-1, 1)).squeeze(-1)                                # (batch_size,)
        with torch.no_grad():
            best_actions = self.model(new_state).argmax(dim=-1)
            new_state_values = self.target_model(new_state).gather(1, best_actions.view(-1, 1)).squeeze(-1)
            target = reward + self.gamma * (1-is_done) * new_state_values                                   # (batch_size,)
        
        # Update model
        self.optim.zero_grad()
        loss = self.criterion(target, output)
        loss.backward()
        self.optim.step()

        # Decay epsilon
        if self.epsilon > self.eps_min:
            self.epsilon *= self.eps_decay_rate
        else:
            self.epsilon = self.eps_min

        return {'Loss': loss.item()}
