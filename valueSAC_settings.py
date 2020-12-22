import numpy as np
from collections import deque
import random
import pygame
import math
import torch
import torch.nn as nn
from torch.distributions import Normal
import torch.nn.functional as F
import special_pid as opt


class Settings():

    def __init__(self):
        self.screen_width = 1920
        self.screen_height = 300
        self.bg_color = (0,192,255)
        self.title = 'double order inverted pendulum'

        self.EPISODE = 10000
        self.TEST = 2
        self.TEST1 = 10
        self.STEP = 1000
        self.max_action = 2

        self.matG = np.array([[1, 0, 0.01, 0], [0, 1.0019, 0, 0.01], [0, 0, 1, 0], [0, 0.3752, 0, 1.0019]])
        self.matH = np.array([0.0001, -0.0002, 0.01, -0.0375])
        self.noise = np.array([0.001, 0.01, 0.01, 0.1])

class Car():
    def __init__(self, screen):
        self.screen = screen
        self.image = pygame.image.load('car.png')
        self.rect = self.image.get_rect()
        self.screen_rect = self.screen.get_rect()
        self.rect.centerx = self.screen_rect.centerx
        self.rect.bottom = 270
        self.pendulum_length = 200

    def xshift(self, xposition):
        self.rect.centerx = self.screen_rect.centerx + int(100 * xposition)
        self.screen.blit(self.image, self.rect)

    def draw_pendulum(self, screen, xposition, arc):
        bottomx = self.screen_rect.centerx + int(100 * xposition)
        bottomy = 250
        topx = int(bottomx + math.sin(arc) * self.pendulum_length)
        topy = int(bottomy - math.cos(arc) * self.pendulum_length)
        pygame.draw.line(screen, (255, 128, 0), (bottomx, bottomy), (topx, topy), 10)
        pygame.draw.circle(screen, (192, 0, 192), (bottomx + 1, bottomy), 6, 0)


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
        self.replay_memory_store = deque()
        self.step_index = 0
        self.replace = 500
        self.replace_sign = replace_sign
        self.replace_rate = 0.01
        self.gamma = 0.97
        self.memory_size = 10000
        self.batch_size = 32
        self.alpha = 0.1

        self.test_sign = 0
        self.end_sign = 0
        self.state_dim = state_dim
        self.action_dim = action_dim

        self.policy_net = policy_net(state_dim, action_dim).cuda()
        self.value_net = value_net(state_dim).cuda()
        self.value_target = value_net(state_dim).cuda()
        self.value_target.training = False
        self.Qvalue_net1 = Qvaluenet(state_dim, action_dim).cuda()
        self.Qvalue_net2 = Qvaluenet(state_dim, action_dim).cuda()

        self.nets = [self.policy_net, self.value_net, self.Qvalue_net1, self.Qvalue_net2]


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

            if self.oldnet_sign == 1:
                onew_action_batch, onew_entropy_batch = self.policy_old.evaluate(state_batch)
                onew_Qvalue1_batch = self.Qvalue_old1(state_batch, onew_action_batch)
                onew_Qvalue2_batch = self.Qvalue_old2(state_batch, onew_action_batch)
                onew_Qvalue_batch = torch.min(onew_Qvalue1_batch,onew_Qvalue2_batch)
                ovalue_batch = self.value_old(state_batch)
                oQvalue1_batch = self.Qvalue_old1(state_batch, action_batch)
                oQvalue2_batch = self.Qvalue_old2(state_batch, action_batch)
                ovalue_loss = F.mse_loss(ovalue_batch, (onew_Qvalue_batch - self.alpha * onew_entropy_batch).detach())
                opolicy_loss = torch.mean(self.alpha * onew_entropy_batch - onew_Qvalue_batch)
                oQvalue1_loss = F.mse_loss(oQvalue1_batch, (reward_batch + self.gamma * next_value_batch).detach())
                oQvalue2_loss = F.mse_loss(oQvalue2_batch, (reward_batch + self.gamma * next_value_batch).detach())
                ovalue_loss.backward()
                opolicy_loss.backward()
                oQvalue1_loss.backward()
                oQvalue2_loss.backward()
                parameters = list(self.value_old.parameters())
                old_grads = [parameter.grad.clone() for parameter in parameters]
                self.value_optimizer.get_oldgrad(old_grads)
                parameters = list(self.policy_old.parameters())
                old_grads = [parameter.grad.clone() for parameter in parameters]
                self.policy_optimizer.get_oldgrad(old_grads)
                parameters = list(self.Qvalue_old1.parameters())
                old_grads = [parameter.grad.clone() for parameter in parameters]
                self.Qvalue_optimizer1.get_oldgrad(old_grads)
                parameters = list(self.Qvalue_old2.parameters())
                old_grads = [parameter.grad.clone() for parameter in parameters]
                self.Qvalue_optimizer2.get_oldgrad(old_grads)
                torch.save(self.policy_net, 'data/nets/policy.pkl')
                torch.save(self.value_net, 'data/nets/value.pkl')
                torch.save(self.Qvalue_net1, 'data/nets/Qvalue1.pkl')
                torch.save(self.Qvalue_net2, 'data/nets/Qvalue2.pkl')
                self.policy_old = torch.load('data/nets/policy.pkl')
                self.value_old = torch.load('data/nets/value.pkl')
                self.Qvalue_old1 = torch.load('data/nets/Qvalue1.pkl')
                self.Qvalue_old2 = torch.load('data/nets/Qvalue2.pkl')

            self.policy_optimizer.zero_grad()
            policy_loss.backward()
            self.policy_optimizer.step()

            self.value_optimizer.zero_grad()
            value_loss.backward()
            self.value_optimizer.step()

            self.Qvalue_optimizer1.zero_grad()
            Qvalue1_loss.backward()
            self.Qvalue_optimizer1.step()

            self.Qvalue_optimizer2.zero_grad()
            Qvalue2_loss.backward()
            self.Qvalue_optimizer2.step()

    def get_task_message(self):
        self. algorithms = ['0.Adam', '1.RMSprop', '2.single_Adapid', '3.double_Adapid', '4.PID', '5.Adam_origin', '6.SGD-momentum']
        self.derivative_sign = [0, 0, 1, 1, 1, 0, 0] #是否带微分项标志，与上方的算法对应
        self.Adaptive_sign = [1, 1, 1, 1, 0, 1, 0]#是否带自适应项标志
        task = int(input('please input a task. 0 for algorithm modify, 1 for learning rate modify, 2 for beta modify, 3 for PID modify: \n ' ))
        if task == 0:
            algorithms = eval(input('please input the list of test algorithms: \n'))
            algorithms = [int(i) for i in algorithms]
            self.test_len = len(algorithms)
        else:
            algorithms = int(input('please input a single algorithm: \n'))
        if task == 1 or task == 0:
            lrs = eval(input('please input the list of learning rates: \n'))
            if task == 0:
                if len(lrs) > len(algorithms):
                    lrs = lrs[:len(algorithms)]
                else:
                    for i in range(len(algorithms) - len(lrs)):
                        lrs.append(lrs[-1])
            self.test_len = len(lrs)
        else:
            lrs = float(input('please input a single learning rate: \n'))
        if task == 2 or task == 0:
            betas = eval(input(('please input the list of testing betas: \n')))
            if task == 0:
                if len(betas) > len(algorithms):
                    betas = betas[:len(algorithms)]
                else:
                    for i in range(len(algorithms) - len(betas)):
                        betas.append(betas[-1])
            self.test_len = len(betas)
        else:
            betas = float(input('please input a single beta value (list of beta1 and beta2 for Adaptive algorithm):\n'))
        if task == 0:
            PIparameter = eval(input('please input the list of testing pid parameters:'))
            if len(PIparameter) > len(algorithms):
                PIparameter = PIparameter[:len(algorithms)]
            else:
                for i in range(len(algorithms) - len(PIparameter)):
                    betas.append(PIparameter[-1])
        elif self.derivative_sign[algorithms] == 1:
            if task == 3:
                PIparameter = eval(input('please input the list of testing pid parameters:'))
                self.test_len = len(PIparameter)
            else:
                PIparameter = float(input('please input a single pid parameter group value: \n'))
        else:
            PIparameter = []
        self.task = task
        self.algorithms = algorithms
        self.lrs = lrs
        self.betas = betas
        self.PIparameter = PIparameter

    def set_optimizers(self):
        optimizers = []
        self.oldnet_sign = 0
        for net in self.nets:
            if self.current_algorithm == 0:
                optimizer = opt.Adamoptimizer(net.parameters(), lr=self.current_lr, weight_decay=0.0001,
                                              momentum=self.current_beta[0], beta=self.current_beta[1])
            elif self.current_algorithm == 1:
                optimizer = opt.RMSpropoptimizer(net.parameters(), lr=self.current_lr, weight_decay=0.0001, beta=self.current_beta)
            elif self.current_algorithm == 2:
                optimizer = opt.Adapidoptimizer(net.parameters(), lr=self.current_lr, weight_decay=0.0001,
                                                momentum=self.current_beta[0], beta=self.current_beta[1],
                                                I=self.current_PIparameter[0], D=self.current_PIparameter[1])
            if self.current_PIparameter[1] != 0:
                self.oldnet_sign = 1
            elif self.current_algorithm == 3:
                optimizer = opt.double_Adapidoptimizer(net.parameters(), lr=self.current_lr, weight_decay=0.0001,
                                                momentum=self.current_beta[0], beta=self.current_beta[1],
                                                I=self.current_PIparameter[0], D=self.current_PIparameter[1])
            if self.current_PIparameter[1] != 0:
                self.oldnet_sign = 1
            elif self.current_algorithm == 4:
                optimizer = opt.PIDoptimizer(net.parameters(), lr=self.current_lr, weight_decay=0.0001,
                                                momentum=self.current_beta,
                                                I=self.current_PIparameter[0], D=self.current_PIparameter[1])
                if self.current_PIparameter[1] != 0:
                    self.oldnet_sign = 1
            elif self.current_algorithm == 5:
                optimizer = torch.optim.Adam(net.parameters(), lr=self.current_lr,
                                             betas=(self.current_beta[0], self.current_beta[1]))
            elif self.current_algorithm == 6:
                optimizer = torch.optim.SGD(net.parameters(), lr=self.current_lr, momentum=self.current_beta)
            else:
                raise ValueError('Not correct algorithm symbol')
            optimizers.append(optimizer)
        self.policy_optimizer = optimizers[0]
        self.value_optimizer = optimizers[1]
        self.Qvalue_optimizer1 = optimizers[2]
        self.Qvalue_optimizer2 = optimizers[3]
        if self.oldnet_sign == 1:
            torch.save(self.policy_net, 'data/nets/policy.pkl')
            torch.save(self.value_net, 'data/nets/value.pkl')
            torch.save(self.Qvalue_net1, 'data/nets/Qvalue1.pkl')
            torch.save(self.Qvalue_net2, 'data/nets/Qvalue2.pkl')
            self.policy_old = torch.load('data/nets/policy.pkl')
            self.value_old = torch.load('data/nets/value.pkl')
            self.Qvalue_old1 = torch.load('data/nets/Qvalue1.pkl')
            self.Qvalue_old2 = torch.load('data/nets/Qvalue2.pkl')

    def set_current_parameters(self):
        if self.test_sign >= self.test_len:
            self.end_sign = 1
        else:
            if self.task == 0:
                self.current_algorithm = self.algorithms[self.test_sign]
            else:
                self.current_algorithm = self.algorithms
            if self.task == 0 or self.task == 1:
                self.current_lr = self.lrs[self.test_sign]
            else:
                self.current_lr = self.lrs
            if self.task == 0 or self.task == 2:
                self.current_beta = self.betas[self.test_sign]
            else:
                self.current_beta = self.betas
            if self.Adaptive_sign[self.current_algorithm] == 0 and isinstance(self.current_beta, list) == True:
                self.current_beta = self.current_beta[0]
            elif self.Adaptive_sign[self.current_algorithm] == 1 and isinstance(self.current_beta, list) == False:
                self.current_beta = [self.current_beta, self.current_beta]
            if self.derivative_sign[self.current_algorithm] == True:
                if self.task == 0 or self.task == 3:
                    self.current_PIparameter = self.PIparameter[self.test_sign]
                else:
                    self.current_PIparameter = self.PIparameter
                if isinstance(self.current_PIparameter, list) == False:
                    self.current_PIparameter = [self.current_PIparameter, 0]
        self.test_sign += 1

    def task_initialize(self):
        self.replay_memory_store = deque()
        self.step_index = 0
        self.policy_net = policy_net(self.state_dim, self.action_dim).cuda()
        self.value_net = value_net(self.state_dim).cuda()
        self.value_target = value_net(self.state_dim).cuda()
        self.value_target.training = False
        self.Qvalue_net1 = Qvaluenet(self.state_dim, self.action_dim).cuda()
        self.Qvalue_net2 = Qvaluenet(self.state_dim, self.action_dim).cuda()
        self.nets = [self.policy_net, self.value_net, self.Qvalue_net1, self.Qvalue_net2]














