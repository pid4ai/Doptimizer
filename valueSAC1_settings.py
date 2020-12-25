import numpy as np
from collections import deque
import random
import pygame
import math
import torch
import torch.nn as nn
from torch.distributions import Normal
import torch.nn.functional as F


class Settings():

    def __init__(self):
        self.screen_width = 1360
        self.screen_height = 400
        self.bg_color = (0,192,255)
        self.title = 'double order inverted pendulum'

        self.EPISODE = 10000
        self.TEST = 2
        self.TEST1 = 10
        self.STEP = 5000
        self.max_action = 2

        self.matG = np.array([[1, 0, 0, 0.01, 0, 0], [0, 1.0032, -0.0016, 0, 0.01, 0], [0, -0.0048, 1.0043, 0, 0, 0.01],
                              [0, 0, 0, 1, 0, 0], [0, 0.6441, -0.3222, 0, 1.0032, -0.0016],
                              [0, -0.9667, 0.8589, 0, -0.0048, 1.0043]])
        self.matH = np.array([0.0001, -0.0002, 0.0001, 0.01, -0.0322, 0.0108])

class Car():
    def __init__(self,screen):
        self.screen = screen
        self.image = pygame.image.load('car.png')
        self.rect = self.image.get_rect()
        self.screen_rect = self.screen.get_rect()
        self.rect.centerx = self.screen_rect.centerx
        self.rect.bottom = 370
        self.pendulum_length = 120

    def xshift(self,xposition):
        self.rect.centerx = self.screen_rect.centerx+int(200*xposition)
        self.screen.blit(self.image, self.rect)

    def draw_pendulum(self,screen,xposition,arc1,arc2):
        bottomx1 = self.screen_rect.centerx+int(200*xposition)
        bottomy1 = 350
        bottomx2 = int(bottomx1+math.sin(arc1)*self.pendulum_length)
        bottomy2 = int(bottomy1-math.cos(arc1)*self.pendulum_length)
        topx = bottomx2+int(math.sin(arc2)*self.pendulum_length)
        topy = bottomy2-int(math.cos(arc2)*self.pendulum_length)
        pygame.draw.line(screen, (255, 128, 0), (bottomx1, bottomy1), (bottomx2, bottomy2), 10)
        pygame.draw.line(screen, (255, 128, 0), (bottomx2, bottomy2), (topx, topy), 10)
        pygame.draw.circle(screen, (192, 0, 192), (bottomx1 + 1, bottomy1), 6, 0)
        pygame.draw.circle(screen,(192,0,192),(bottomx2+1,bottomy2),6,0)


class policy_net(nn.Module):
    def __init__(self, state_dim, action_dim, normal = 10 ** -4):
        super(policy_net, self).__init__()
        self.action_dim = action_dim
        self.layer1 = nn.Linear(state_dim, 128)
        self.layer2 = nn. Linear(128, 128)
        self.mean_layer = nn.Linear(128, action_dim)
        self.std_layer = nn.Linear(128, action_dim)
        self.relu = nn.ReLU()
        self.mean_layer.weight.data.uniform_(-normal, normal)
        self.mean_layer.bias.data.uniform_(-normal, normal)
        self.std_layer.weight.data.uniform_(-normal, normal)
        self.std_layer.bias.data.uniform_(-normal, normal)

    def forward(self, state):
        state = self.layer1(state)
        state = self.relu(state)
        state = self.layer2(state)
        state = self.relu(state)
        mean = self.mean_layer(state)
        log_std = torch.clamp(self.std_layer(state) - 1, -20, 1)
        return mean, log_std

    def get_action(self, state):
        state = torch.from_numpy(state).cuda().float()
        mean, log_std = self.forward(state)
        mean = mean.detach().cpu().numpy()
        std = np.exp(log_std.detach().cpu().numpy())
        action = np.random.normal(mean, std)
        return action

    def evaluate(self, state):
        mean, log_std = self.forward(state)
        std = log_std.exp()
        distribution = Normal(mean, std)
        rand = torch.from_numpy(np.random.normal(0, 1, size = std.size())).cuda().float()
        action = mean + std * rand.cuda()
        log_prob = distribution.log_prob(action).cuda()
        return action, log_prob


class value_net(nn.Module):
    def __init__(self, state_dim, normal = 10 ** -3):
        super(value_net, self).__init__()
        self.layer1 = nn.Linear(state_dim, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, 1)
        self.relu = nn.ReLU()
        self.layer3.weight.data.uniform_(-normal, normal)
        self.layer3.bias.data.uniform_(-normal, normal)

    def forward(self, state):
        state = self.layer1(state)
        state = self.relu(state)
        state = self.layer2(state)
        state = self.relu(state)
        value = self.layer3(state)
        return value

class Qvaluenet(nn.Module):
    def __init__(self, state_dim, action_dim, normal = 10 ** -3):
        super(Qvaluenet, self).__init__()
        self.layer1 = nn.Linear(state_dim + action_dim, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, 1)
        self.relu = nn.ReLU()
        self.layer3.weight.data.uniform_(-normal, normal)
        self.layer3.bias.data.uniform_(-normal, normal)

    def forward(self, state, action):
        input = torch.cat([state, action], dim = 1)
        input = self.layer1(input)
        input = self.relu(input)
        input = self.layer2(input)
        input = self.relu(input)
        Qvalue = self.layer3(input)
        return Qvalue


class SAC(nn.Module):
    def __init__(self, state_dim, action_dim, replace_sign=1):
        super(SAC, self).__init__()
        self.algorithm = int(input('SGD0 OR ADAM1 \n'))
        if self.algorithm == 0:
            self.optparameter = eval(input('policy SGD parameters, lr, momentum \n'))
        else:
            self.optparameter = eval(input('policy Adam parameters, lr, beta \n'))
        self.valgorithm = int(input('SGD0 OR ADAM 1 \n'))
        if self.valgorithm == 0:
            self.voptparameter = eval(input('value SGD parameters, lr ,momentum\n'))
        else:
            self.voptparameter = eval(input('value Adam parameters, lr, beta\n'))
        self.replay_memory_store = deque()
        self.step_index = 0
        self.replace = 500
        self.replace_sign = replace_sign
        self.replace_rate = 0.01
        #self.action_lr = 0.00002
        #self.value_lr = float(input('please input value_lr \n'))
        self.action_renew_steps = 5
        self.gamma = 0.97
        self.memory_size = 30000
        self.batch_size = 32
        self.alpha = 0.2
        self.train_steps = 0

        self.policy_net = policy_net(state_dim, action_dim).cuda()
        self.value_net = value_net(state_dim).cuda()
        self.value_target = value_net(state_dim).cuda()
        self.value_target.training = False
        self.Qvalue_net1 = Qvaluenet(state_dim, action_dim).cuda()
        self.Qvalue_net2 = Qvaluenet(state_dim, action_dim).cuda()

        if self.algorithm == 0:
            self.policy_optimizer = torch.optim.SGD(self.policy_net.parameters(), lr=self.optparameter[0],
                                                    momentum=self.optparameter[1], weight_decay=0.0001)
        else:
            self.policy_optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=self.optparameter[0],
                                                     betas=(self.optparameter[1][0], self.optparameter[1][1]),
                                                     weight_decay=0.0001)
        if self.valgorithm == 0:
            self.value_optimizer = torch.optim.SGD(self.value_net.parameters(), lr=self.voptparameter[0],
                                                   momentum=self.voptparameter[1], weight_decay=0.0001)
            self.Qvalue_optimizer1 = torch.optim.SGD(self.Qvalue_net1.parameters(), lr=self.voptparameter[0],
                                                     momentum=self.voptparameter[1], weight_decay=0.0001)
            self.Qvalue_optimizer2 = torch.optim.SGD(self.Qvalue_net2.parameters(), lr=self.voptparameter[0],
                                                     momentum=self.voptparameter[1], weight_decay=0.0001)
        else:
            self.value_optimizer = torch.optim.Adam(self.value_net.parameters(), lr=self.voptparameter[0],
                                                    betas=(self.voptparameter[1][0], self.voptparameter[1][1]),
                                                    weight_decay=0.0001)
            self.Qvalue_optimizer1 = torch.optim.Adam(self.Qvalue_net1.parameters(), lr=self.voptparameter[0],
                                                      betas=(self.voptparameter[1][0], self.voptparameter[1][1]),
                                                      weight_decay=0.0001)
            self.Qvalue_optimizer2 = torch.optim.Adam(self.Qvalue_net2.parameters(), lr=self.voptparameter[0],
                                                      betas=(self.voptparameter[1][0], self.voptparameter[1][1]),
                                                      weight_decay=0.0001)

    def SAC_training(self, state, action, reward, next_state, done):
        self.replay_memory_store.append([state, action, reward, next_state, done])
        if len(self.replay_memory_store) > self.memory_size:
            self.replay_memory_store.popleft()
        if len(self.replay_memory_store) > self.batch_size:
            if self.replace_sign == True:
                '''软更新'''
                value_net = list(self.value_net.parameters())
                value_target = list(self.value_target.parameters())
                for i in range(len(value_net)):
                    value_target[i].data.copy_(
                        value_net[i].data.detach() * self.replace_rate + value_target[i].data.detach() * (
                                1 - self.replace_rate))
            elif self.step_index % self.replace == 0:
                '''硬更新'''
                self.value_target.load_state_dict(self.value_net.state_dict())
            if self.step_index % self.replace == 0:
                '''硬更新完成标志，软更新下只是为了监测运行方便点'''
                print('replace completed')
            self.step_index += 1
            batch_label = random.sample(range(len(self.replay_memory_store)), self.batch_size)
            mini_batch = []
            for i in range(self.batch_size):
                mini_batch.append(self.replay_memory_store[batch_label[i]])
            state_batch = torch.from_numpy(np.array([data[0] for data in mini_batch])).cuda().float()
            action_batch = torch.from_numpy(np.array([data[1] for data in mini_batch])).cuda().float()
            reward_batch = torch.from_numpy(np.array([[data[2]] for data in mini_batch])).cuda().float()
            next_state_batch = torch.from_numpy(np.array([data[3] for data in mini_batch])).cuda().float()
            done_batch = torch.from_numpy(np.array([[data[4]] for data in mini_batch])).cuda().float()

            next_value_batch = self.value_target(next_state_batch) * (1 - done_batch)
            new_action_batch, new_entropy_batch = self.policy_net.evaluate(state_batch)
            new_Qvalue1_batch = self.Qvalue_net1(state_batch, new_action_batch)
            new_Qvalue2_batch = self.Qvalue_net2(state_batch, new_action_batch)
            new_Qvalue_batch = torch.min(new_Qvalue1_batch, new_Qvalue2_batch)
            value_batch = self.value_net(state_batch)
            Qvalue1_batch = self.Qvalue_net1(state_batch, action_batch)
            Qvalue2_batch = self.Qvalue_net2(state_batch, action_batch)

            value_loss = F.mse_loss(value_batch, (new_Qvalue_batch - self.alpha * new_entropy_batch).detach())
            policy_loss = torch.mean(self.alpha * new_entropy_batch - new_Qvalue_batch)
            Qvalue1_loss = F.mse_loss(Qvalue1_batch, (reward_batch + self.gamma * next_value_batch).detach())
            Qvalue2_loss = F.mse_loss(Qvalue2_batch, (reward_batch + self.gamma * next_value_batch).detach())

            self.policy_optimizer.zero_grad()
            policy_loss.backward()
            parameters = list(self.policy_net.parameters())
            for parameter in parameters:
                maxgrad = torch.max(torch.abs(parameter.grad))
                if maxgrad > 0.05:
                    parameter.grad = parameter.grad / (maxgrad/0.05)
                #torch.clamp(parameter.grad, -0.05, 0.05)
            self.policy_optimizer.step()

            self.value_optimizer.zero_grad()
            value_loss.backward()
            parameters = list(self.value_net.parameters())
            for parameter in parameters:
                maxgrad = torch.max(torch.abs(parameter.grad))
                if maxgrad > 0.1:
                    parameter.grad = parameter.grad / (maxgrad/0.1)
                #torch.clamp(parameter.grad, -0.1, 0.1)
            self.value_optimizer.step()

            self.Qvalue_optimizer1.zero_grad()
            Qvalue1_loss.backward()
            parameters = list(self.Qvalue_net1.parameters())
            for parameter in parameters:
                maxgrad = torch.max(torch.abs(parameter.grad))
                if maxgrad > 0.1:
                    parameter.grad = parameter.grad / (maxgrad/0.1)
                #torch.clamp(parameter.grad, -0.1, 0.1)
            self.Qvalue_optimizer1.step()

            self.Qvalue_optimizer2.zero_grad()
            Qvalue2_loss.backward()
            parameters=list(self.Qvalue_net2.parameters())
            for parameter in parameters:
                maxgrad = torch.max(torch.abs(parameter.grad))
                if maxgrad > 0.1:
                    parameter.grad = parameter.grad / (maxgrad/0.1)
                #torch.clamp(parameter.grad, -0.1, 0.1)
            self.Qvalue_optimizer2.step()
            self.train_steps += 1
            return -policy_loss.detach().cpu().numpy(), Qvalue1_loss.detach().cpu().numpy()
        else:
            return 0












