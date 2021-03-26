
""" 
Network definitions
"""

import torch
import torch.nn as nn
import numpy as np 

NUM_FRAMES = 1

class ReplayMemory:

    def __init__(self, memory_size, batch_size, resolution, device):
        self.mem_size = memory_size
        self.batch_size = batch_size
        self.device = device
        self.ptr = 0
        res_x, res_y = resolution

        # Tensors to hold state variables
        self.state = torch.FloatTensor(memory_size, NUM_FRAMES, res_x, res_y).zero_().to(self.device)
        self.new_state = torch.FloatTensor(memory_size, NUM_FRAMES, res_x, res_y).zero_().to(self.device)
        self.action = torch.LongTensor(memory_size).zero_().to(self.device)
        self.reward = torch.FloatTensor(memory_size).zero_().to(self.device)
        self.done = torch.LongTensor(memory_size).zero_().to(self.device)

    def add_transaction(self, state, action, reward, new_state, is_done):
        self.state[self.ptr, :, :, :] = state 
        self.new_state[self.ptr, :, :, :] = new_state
        self.action[self.ptr] = action 
        self.reward[self.ptr] = reward 
        self.done[self.ptr] = is_done
    
        self.ptr += 1
        if self.ptr >= self.mem_size:
            self.ptr = 0

    def get_transactions(self):
        choices = np.random.choice(np.arange(self.mem_size), size=self.batch_size, replace=False)
        choices = torch.LongTensor(choices).to(self.device)
        return (
            self.state[choices], self.action[choices], self.reward[choices], 
            self.new_state[choices], self.done[choices]
        )


class DQN(nn.Module):
    """ Resnet 18 with some modifications for small images """

    def __init__(self, action_space_size):
        super().__init__()
        self.conv1 = nn.Conv2d(NUM_FRAMES, 8, kernel_size=3, stride=2, bias=False)
        self.bn1 = nn.BatchNorm2d(8)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(8, 8, kernel_size=3, stride=2, bias=False)
        self.bn2 = nn.BatchNorm2d(8)
        self.relu2 = nn.ReLU()
        self.conv3 = nn.Conv2d(8, 8, kernel_size=3, stride=1, bias=False)            
        self.bn3 = nn.BatchNorm2d(8)
        self.relu3 = nn.ReLU()
        self.conv4 = nn.Conv2d(8, 16, kernel_size=3, stride=1, bias=False)
        self.bn4 = nn.BatchNorm2d(16)
        self.relu4 = nn.ReLU()
        self.fc_1 = nn.Linear(192, 64)     
        self.relu5 = nn.ReLU()                                # (b, action_space_size)
        self.fc_out = nn.Linear(64, action_space_size)

    def forward(self, x):
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        x = self.relu3(self.bn3(self.conv3(x)))
        x = self.relu4(self.bn4(self.conv4(x)))
        x = x.view(-1, 192)
        out = self.fc_out(self.relu5(self.fc_1(x)))
        return out
        

class DuelDQN(nn.Module):
    """ Resnet 18 with some modifications for small images """

    def __init__(self, action_space_size):
        super().__init__()
        self.conv1 = nn.Conv2d(NUM_FRAMES, 8, kernel_size=3, stride=2, bias=False)
        self.bn1 = nn.BatchNorm2d(8)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(8, 8, kernel_size=3, stride=2, bias=False)
        self.bn2 = nn.BatchNorm2d(8)
        self.relu2 = nn.ReLU()
        self.conv3 = nn.Conv2d(8, 8, kernel_size=3, stride=1, bias=False)            
        self.bn3 = nn.BatchNorm2d(8)
        self.relu3 = nn.ReLU()
        self.conv4 = nn.Conv2d(8, 16, kernel_size=3, stride=1, bias=False)
        self.bn4 = nn.BatchNorm2d(16)
        self.relu4 = nn.ReLU()
        self.fc_state = nn.Sequential(
            nn.Linear(96, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )     
        self.fc_advantage = nn.Sequential(
            nn.Linear(96, 64),
            nn.ReLU(),
            nn.Linear(64, action_space_size) 
        )

    def forward(self, x):
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        x = self.relu3(self.bn3(self.conv3(x)))
        x = self.relu4(self.bn4(self.conv4(x)))
        x = x.view(-1, 192)
        x1, x2 = x[:, :96], x[:, 96:]
        state_value = self.fc_state(x1).reshape(-1, 1)
        adv_value = self.fc_advantage(x2)
        out = state_value + (adv_value - adv_value.mean(dim=1).reshape(-1, 1))
        return out


# Testing
if __name__ == "__main__":

    net = DQN(4)
    out = net(torch.rand(1, NUM_FRAMES, 30, 45))
    print(out.size())