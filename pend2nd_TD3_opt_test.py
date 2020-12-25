import numpy as np
from collections import deque
import random
import pygame
import math
import torch
import torch.nn as nn


class Settings():

    def __init__(self):
        self.screen_width = 1080
        self.screen_height = 300
        self.bg_color = (0,192,255)
        self.title = 'double order inverted pendulum'

        self.EPISODE = 4000
        self.TEST = 2
        self.TEST1 = 10
        self.STEP = 2000

        self.matG = np.array([[1, 0, 0, 0.01, 0, 0], [0, 1.0032, -0.0016, 0, 0.01, 0], [0, -0.0048, 1.0043, 0, 0, 0.01],
                              [0, 0, 0, 1, 0, 0], [0, 0.6441, -0.3222, 0, 1.0032, -0.0016],
                              [0, -0.9667, 0.8589, 0, -0.0048, 1.0043]])
        self.matH = np.array([0.0001, -0.0002, 0.0001, 0.01, -0.0322, 0.0108])
        self.noise = np.array([0.001, 0.01, 0.01, 0.01, 0.1, 0.1])

class Car():
    def __init__(self,screen):
        self.screen = screen
        self.image = pygame.image.load('car.png')
        self.rect = self.image.get_rect()
        self.screen_rect = self.screen.get_rect()
        self.rect.centerx = self.screen_rect.centerx
        self.rect.bottom = 270
        self.pendulum_length = 100

    def xshift(self,xposition):
        self.rect.centerx = self.screen_rect.centerx+int(500*xposition)
        self.screen.blit(self.image, self.rect)

    def draw_pendulum(self,screen,xposition,arc1,arc2):
        bottomx1 = self.screen_rect.centerx+int(500*xposition)
        bottomy1 = 250
        bottomx2 = int(bottomx1+math.sin(arc1)*self.pendulum_length)
        bottomy2 = int(bottomy1-math.cos(arc1)*self.pendulum_length)
        topx = bottomx2+int(math.sin(arc2)*self.pendulum_length)
        topy = bottomy2-int(math.cos(arc2)*self.pendulum_length)
        pygame.draw.line(screen, (255, 128, 0), (bottomx1, bottomy1), (bottomx2, bottomy2), 10)
        pygame.draw.line(screen, (255, 128, 0), (bottomx2, bottomy2), (topx, topy), 10)
        pygame.draw.circle(screen, (192, 0, 192), (bottomx1 + 1, bottomy1), 6, 0)
        pygame.draw.circle(screen,(192,0,192),(bottomx2+1,bottomy2),6,0)


class DeepQNetwork(nn.Module):


    def __init__(self, max_action, soft_replace=True, optim_algorithm = 0):
        super(DeepQNetwork,self).__init__()
        '''超参数设定'''
        self.step_index = 0
        self.replace = 500
        self.replay_memory_store = deque()
        self.action_lr = 0.001
        self.state_lr = 0.001
        self.model_lr = 0.01
        self.action_renew_steps = 5
        self.gamma = 0.97
        self.memory_size = 10000
        self.batch_size = 64
        self.initial_epsilon = 1
        self.final_epsilon = 0.1
        self.epsilon = self.initial_epsilon  # epsilon_greedy-policy
        self.explore = 10000
        self.max_action = max_action
        self.soft_replace = soft_replace
        self.replace_rate = 0.01
        self.optim_algorithm = optim_algorithm

        self.modelloss = []

        '''Optimizer_hyperparameters'''
        self.beta = [0.9,0.99]
        self.current_statebeta = [1,1]
        self.current_actionbeta = [1,1]
        self.RMSbeta = 0.99
        self.kp = 0.5
        self.ki = 1
        self.kd = 0
        self.RMSepsilon = 0.00000001
        '''表格Q学习参数设定'''
        self.Qchart  = np.array([])

        self.state_dim = 6
        self.network_initialize()
        self.create_optimizers()


    '''搭建神经网络及设置误差'''
    def network_initialize(self):

        self.action_eval = nn.Sequential(nn.Linear(self.state_dim, 50),
                                    nn.ReLU(),
                                    nn.Linear(50, 1),
                                    nn.Tanh()).cuda()
        parameters = list(self.action_eval.parameters())
        torch.nn.init.normal_(parameters[0],0,0.02)
        torch.nn.init.constant_(parameters[1], 0)
        torch.nn.init.normal_(parameters[2], 0,0.02)
        torch.nn.init.constant_(parameters[3], 0)
        self.action_target = nn.Sequential(nn.Linear(self.state_dim, 50),
                                    nn.ReLU(),
                                    nn.Linear(50, 1),
                                    nn.Tanh()).cuda()
        parameters = list(self.action_target.parameters())
        torch.nn.init.normal_(parameters[0], 0, 0.02)
        torch.nn.init.constant_(parameters[1], 0)
        torch.nn.init.normal_(parameters[2], 0, 0.02)
        torch.nn.init.constant_(parameters[3], 0)
        self.state_eval1 = nn.Sequential(nn.Linear(self.state_dim+1, 50),
                                        nn.ReLU(),
                                        nn.Linear(50,1),
                                        nn.ReLU()).cuda()
        parameters = list(self.state_eval1.parameters())
        torch.nn.init.normal_(parameters[0], 0, 0.02)
        torch.nn.init.constant_(parameters[1], 0.01)
        torch.nn.init.normal_(parameters[2], 0, 0.02)
        torch.nn.init.constant_(parameters[3], 0.01)
        self.state_eval2 = nn.Sequential(nn.Linear(self.state_dim + 1, 50),
                                        nn.ReLU(),
                                        nn.Linear(50, 1),
                                        nn.ReLU()).cuda()
        parameters = list(self.state_eval2.parameters())
        torch.nn.init.normal_(parameters[0], 0, 0.02)
        torch.nn.init.constant_(parameters[1], 0.01)
        torch.nn.init.normal_(parameters[2], 0, 0.02)
        torch.nn.init.constant_(parameters[3], 0.01)
        self.state_target1 = nn.Sequential(nn.Linear(self.state_dim+1, 50),
                                        nn.ReLU(),
                                        nn.Linear(50,1),
                                        nn.ReLU()).cuda()
        parameters = list(self.state_target1.parameters())
        torch.nn.init.normal_(parameters[0], 0, 0.02)
        torch.nn.init.constant_(parameters[1], 0.01)
        torch.nn.init.normal_(parameters[2], 0, 0.02)
        torch.nn.init.constant_(parameters[3], 0.01)
        self.state_target2 = nn.Sequential(nn.Linear(self.state_dim + 1, 50),
                                          nn.ReLU(),
                                          nn.Linear(50, 1),
                                          nn.ReLU()).cuda()
        parameters = list(self.state_target2.parameters())
        torch.nn.init.normal_(parameters[0], 0, 0.02)
        torch.nn.init.constant_(parameters[1], 0.01)
        torch.nn.init.normal_(parameters[2], 0, 0.02)
        torch.nn.init.constant_(parameters[3], 0.01)
        self.model_net = nn.Sequential(nn.Linear(self.state_dim + 1, 50),
                                       nn.ReLU(),
                                       nn.Linear(50,50),
                                       nn.ReLU(),
                                       nn.Linear(50,6)).cuda()
        '''值网络有2组，分别对应一个loss'''
        self.action_target.training = False
        self.state_target1.training = False
        self.state_target2.training = False

    '''优化器和损失函数'''
    def create_optimizers(self):
        self.model_optimizer = torch.optim.Adam(self.model_net.parameters(), lr=self.model_lr)
        if self.optim_algorithm == 0:
            self.state_optimizer1 = torch.optim.Adam(self.state_eval1.parameters(), lr=self.state_lr)
            self.state_optimizer2 = torch.optim.Adam(self.state_eval2.parameters(), lr=self.state_lr)
            self.action_optimizer = torch.optim.Adam(self.action_eval.parameters(), lr=self.action_lr)
        elif self.optim_algorithm == 1:
            self.state_optimizer1 = torch.optim.RMSprop(self.state_eval1.parameters(), lr=self.state_lr)
            self.state_optimizer2 = torch.optim.RMSprop(self.state_eval2.parameters(), lr=self.state_lr)
            self.action_optimizer = torch.optim.RMSprop(self.action_eval.parameters(), lr=self.action_lr)
        elif self. optim_algorithm == 2:
            self.state_optimizer1 = torch.optim.Adagrad(self.state_eval1.parameters(), lr=self.state_lr)
            self.state_optimizer2 = torch.optim.Adagrad(self.state_eval2.parameters(), lr=self.state_lr)
            self.action_optimizer = torch.optim.Adagrad(self.action_eval.parameters(), lr=self.action_lr)
        else:
            self.state_optimizer1 = torch.optim.SGD(self.state_eval1.parameters(), lr=self.state_lr)
            self.state_optimizer2 = torch.optim.SGD(self.state_eval2.parameters(), lr=self.state_lr)
            self.action_optimizer = torch.optim.SGD(self.action_eval.parameters(), lr=self.action_lr)

        self.loss = nn.MSELoss().cuda()
        '''用于优化器编程的权重储存'''
        self.state_momentum1 = [torch.zeros([50, self.state_dim + 1]).cuda(), torch.zeros(50).cuda(),
                                torch.zeros([1, 50]).cuda(), torch.zeros(1).cuda()]
        self.state_momentum2 = [torch.zeros([50, self.state_dim + 1]).cuda(), torch.zeros(50).cuda(),
                                torch.zeros([1, 50]).cuda(), torch.zeros(1).cuda()]
        self.action_momentum = [torch.zeros([50, self.state_dim]).cuda(), torch.zeros(50).cuda(),
                                torch.zeros([1, 50]).cuda(), torch.zeros(1).cuda()]
        self.state_lastgrad1 = [torch.zeros([50, self.state_dim + 1]).cuda(), torch.zeros(50).cuda(),
                                torch.zeros([1, 50]).cuda(), torch.zeros(1).cuda()]
        self.state_lastgrad2 = [torch.zeros([50, self.state_dim + 1]).cuda(), torch.zeros(50).cuda(),
                                torch.zeros([1, 50]).cuda(), torch.zeros(1).cuda()]
        self.action_lastgrad = [torch.zeros([50, self.state_dim]).cuda(), torch.zeros(50).cuda(),
                                torch.zeros([1, 50]).cuda(), torch.zeros(1).cuda()]
        self.state_vt1 = [torch.zeros([50, self.state_dim + 1]).cuda(), torch.zeros(50).cuda(),
                          torch.zeros([1, 50]).cuda(), torch.zeros(1).cuda()]
        self.state_vt2 = [torch.zeros([50, self.state_dim + 1]).cuda(), torch.zeros(50).cuda(),
                          torch.zeros([1, 50]).cuda(), torch.zeros(1).cuda()]
        self.action_vt = [torch.zeros([50, self.state_dim]).cuda(), torch.zeros(50).cuda(),
                          torch.zeros([1, 50]).cuda(), torch.zeros(1).cuda()]
        self.state_eps1 = [torch.zeros([50, self.state_dim + 1]).cuda(), torch.zeros(50).cuda(),
                           torch.zeros([1, 50]).cuda(), torch.zeros(1).cuda()]
        self.state_eps2 = [torch.zeros([50, self.state_dim + 1]).cuda(), torch.zeros(50).cuda(),
                           torch.zeros([1, 50]).cuda(), torch.zeros(1).cuda()]
        self.action_eps = [torch.zeros([50, self.state_dim]).cuda(), torch.zeros(50).cuda(),
                           torch.zeros([1, 50]).cuda(), torch.zeros(1).cuda()]
        for i in range(len(self.state_eps1)):
            nn.init.constant_(self.state_eps1[i], self.RMSepsilon)
            nn.init.constant_(self.state_eps2[i], self.RMSepsilon)
            nn.init.constant_(self.action_eps[i], self.RMSepsilon)

    '''模型网络真实精确度评价'''
    def model_eval(self, state, next_state, action):
        deltastate = torch.from_numpy((state - next_state)).cuda().float()
        eval_deltastate = self.model_net(torch.from_numpy(np.concatenate([state, action], axis=0)).cuda().float())
        model_loss = self.loss(deltastate, eval_deltastate)
        self.modelloss.append(model_loss.detach().cpu().numpy())



    '''动作函数（集成了探索和最大值两种动作选取方法）'''
    def action(self, state, explore_sign=True):
        if explore_sign == True:
            if self.step_index < 10000:
                self.epsilon -= (self.initial_epsilon - self.final_epsilon) / self.explore
            action = self.action_eval((torch.from_numpy(state).cuda().float())) * self.max_action
            return action.detach().cpu().numpy() + np.random.multivariate_normal([0], [[self.epsilon]])
        else:
            action = self.action_eval((torch.from_numpy(state).cuda().float())) * self.max_action
            return action.detach().cpu().numpy()


    '''接收经验并训练'''
    def perceive(self, state, action, reward, next_state, done):
        if len(self.replay_memory_store) > 5*self.batch_size:
            if self.soft_replace == True:
                '''软更新'''
                action_eval = list(self.action_eval.parameters())
                action_target = list(self.action_target.parameters())
                state_eval1 = list(self.state_eval1.parameters())
                state_target1 = list(self.state_target1.parameters())
                state_eval2 = list(self.state_eval2.parameters())
                state_target2 = list(self.state_target2.parameters())
                for i in range(len(action_eval)):
                    action_target[i].data.copy_(
                        action_eval[i].data.detach() * self.replace_rate + action_target[i].data.detach() * (
                                1 - self.replace_rate))
                for i in range(len(state_eval1)):
                    state_target1[i].data.copy_(
                        state_eval1[i].data.detach() * self.replace_rate + state_target1[i].data.detach() * (
                                1 - self.replace_rate))
                    state_target2[i].data.copy_(
                        state_eval2[i].data.detach() * self.replace_rate + state_target2[i].data.detach() * (
                                1 - self.replace_rate))
            elif self.step_index % self.replace == 0:
                '''硬更新'''
                self.action_target.load_state_dict(self.action_eval.state_dict())
                self.state_target.load_state_dict(self.state_eval.state_dict())
            if self.step_index % self.replace == 0:
                '''硬更新完成标志，软更新下只是为了监测运行方便点'''
                print('replace completed')
            self.step_index += 1
            '''取大样本'''
            batch_label = random.sample(range(len(self.replay_memory_store)), 5 * self.batch_size)
            mini_batch = []
            for i in range(len(batch_label)):
                mini_batch.append(self.replay_memory_store[batch_label[i]])
            state_batch = torch.from_numpy(np.array([data[0] for data in mini_batch])).cuda().float()
            action_batch = torch.from_numpy(np.array([data[1] for data in mini_batch])).cuda().float()
            reward_batch = [data[2] for data in mini_batch]
            next_state_batch = torch.from_numpy(np.array([data[3] for data in mini_batch])).cuda().float()
            y_batch = []
            next_action_batch = self.action_target(next_state_batch) * self.max_action
            Q_val1_batch = self.state_target1(
                torch.cat([next_state_batch, next_action_batch], dim=1)).detach().cpu().numpy()
            for i in range(0, 5 * self.batch_size):
                done = mini_batch[i][4]
                if done:
                    y_batch.append([reward_batch[i]])
                else:
                    y_batch.append([reward_batch[i] + self.gamma * Q_val1_batch[i][0]])
            y_batch = torch.from_numpy(np.array(y_batch)).cuda().float()
            Q_batch = self.state_eval1(torch.cat([state_batch, action_batch], dim=1)).cuda().float()
            '''经验优先回放机制'''
            err = (y_batch - Q_batch).abs().detach().cpu().numpy()
            small_batch_label = [0]
            small_err_batch = [err[0]]
            min_err = err[0]
            min_label = batch_label[0]
            for i in range(1, self.batch_size * 5):
                if i < self.batch_size:
                    small_batch_label.append(i)
                    small_err_batch.append(err[i])
                else:
                    small_min = small_err_batch[0]
                    min_index = 0
                    for j in range(1, self.batch_size):
                        if small_err_batch[j] < small_min:
                            min_index = j
                            small_min = small_err_batch[j]
                    if small_min < err[i]:
                        small_batch_label[min_index] = i
                        small_err_batch[min_index] = err[i]
                if err[i] < min_err:
                    min_err = err[i]
                    min_label = batch_label[i]
            '''替换掉一个最没用的样本'''
            if len(self.replay_memory_store) > self.memory_size:
                self.replay_memory_store[min_label] = [state, action, reward, next_state, done]
            else:
                self.replay_memory_store.append([state, action, reward, next_state, done])
            small_batch = []
            '''筛选出来的小batch'''
            for i in range(self.batch_size):
                small_batch.append(mini_batch[small_batch_label[i]])
            state_batch = torch.from_numpy(np.array([data[0] for data in small_batch])).cuda().float()
            action_batch = torch.from_numpy(np.array([data[1] for data in small_batch])).cuda().float()
            reward_batch = [data[2] for data in small_batch]
            next_state_batch = torch.from_numpy(np.array([data[3] for data in small_batch])).cuda().float()
            y_batch = []
            next_action_batch = self.action_target(next_state_batch) * self.max_action
            '''双网络取小的那一个值'''
            Q_val1_batch = self.state_target1(
                torch.cat([next_state_batch, next_action_batch], dim=1)).detach().cpu().numpy()
            Q_val2_batch = self.state_target2(
                torch.cat([next_state_batch, next_action_batch], dim=1)).detach().cpu().numpy()
            Q_val_batch = np.concatenate([Q_val1_batch, Q_val2_batch], axis=1).min(1)
            for i in range(0, self.batch_size):
                done = mini_batch[i][4]
                if done:
                    y_batch.append([reward_batch[i]])
                else:
                    y_batch.append([reward_batch[i] + self.gamma * Q_val_batch[i]])
            y_batch = torch.from_numpy(np.array(y_batch)).cuda().float()
            Q_batch1 = self.state_eval1(torch.cat([state_batch, action_batch], dim=1)).cuda().float()
            Q_batch2 = self.state_eval2(torch.cat([state_batch, action_batch], dim=1)).cuda().float()
            '''模型网络更新'''
            deltastate_batch = next_state_batch - state_batch
            evalbatch = self.model_net(torch.cat([state_batch, action_batch], dim=1)).cuda().float()
            modelloss = self.loss(deltastate_batch, evalbatch)
            self.model_optimizer.zero_grad()
            modelloss.backward()
            self.model_optimizer.step()
            '''值网络及动作网络更新'''
            if self.optim_algorithm < 4:  # 四种较为常规的优化算法
                loss1 = self.loss(y_batch, Q_batch1).cuda()
                self.state_optimizer1.zero_grad()  # clear gradients for this training step
                loss1.backward()  # backpropagation, compute gradients
                self.state_optimizer1.step()
                '''值网络2更新'''
                loss2 = self.loss(y_batch, Q_batch2).cuda()
                self.state_optimizer2.zero_grad()  # clear gradients for this training step
                loss2.backward()  # backpropagation, compute gradients
                self.state_optimizer2.step()
                '''动作网络更新'''
                if self.step_index % self.action_renew_steps == 0:
                    if random.uniform(0, 1) < 0.5:
                        new_action = (self.action_eval(state_batch) * self.max_action).float()
                        Q_val = self.state_eval1(torch.cat([state_batch, new_action], dim=1))
                        loss3 = (-Q_val.sum() / self.batch_size)
                        self.action_optimizer.zero_grad()
                        loss3.backward()
                        self.action_optimizer.step()
                    else:
                        new_action = (self.action_eval(state_batch) * self.max_action).float()
                        Q_val = self.state_eval2(torch.cat([state_batch, new_action], dim=1))
                        loss3 = (-Q_val.sum() / self.batch_size)
                        self.action_optimizer.zero_grad()
                        loss3.backward()
                        self.action_optimizer.step()
            elif self.optim_algorithm == 4:
                '''Momentum'''
                self.current_statebeta[0] *= self.beta[0]
                loss1 = self.loss(y_batch, Q_batch1).cuda()
                self.state_optimizer1.zero_grad()
                loss1.backward()
                state_eval1 = list(self.state_eval1.parameters())
                for i in range(len(state_eval1)):
                    self.state_momentum1[i] = self.state_momentum1[i] * self.beta[0] + state_eval1[i].grad * (
                                1 - self.beta[0])
                    state_eval1[i].grad = self.state_momentum1[i] / (1 - self.current_statebeta[0])
                self.state_optimizer1.step()
                loss2 = self.loss(y_batch, Q_batch2).cuda()
                self.state_optimizer2.zero_grad()
                loss2.backward()
                state_eval2 = list(self.state_eval2.parameters())
                for i in range(len(state_eval2)):
                    self.state_momentum2[i] = self.state_momentum2[i] * self.beta[0] + state_eval2[i].grad * (
                                1 - self.beta[0])
                    state_eval2[i].grad = self.state_momentum2[i] / (1 - self.current_statebeta[0])
                self.state_optimizer2.step()
                if self.step_index % self.action_renew_steps == 0:
                    self.current_actionbeta[0] *= self.beta[0]
                    if random.uniform(0, 1) < 0.5:
                        new_action = (self.action_eval(state_batch) * self.max_action).float()
                        Q_val = self.state_eval1(torch.cat([state_batch, new_action], dim=1))
                    else:
                        new_action = (self.action_eval(state_batch) * self.max_action).float()
                        Q_val = self.state_eval2(torch.cat([state_batch, new_action], dim=1))
                    loss3 = (-Q_val.sum() / self.batch_size)
                    self.action_optimizer.zero_grad()
                    loss3.backward()
                    action_eval = list(self.action_eval.parameters())
                    for i in range(len(action_eval)):
                        self.action_momentum[i] = self.action_momentum[i] * self.beta[0] + action_eval[i].grad * (
                                1 - self.beta[0])
                        action_eval[i].grad = self.action_momentum[i] / (1 - self.current_actionbeta[0])
                    self.action_optimizer.step()
            elif self.optim_algorithm == 5:
                '''PID'''
                self.current_statebeta[0] *= self.beta[0]
                loss1 = self.loss(y_batch, Q_batch1).cuda()
                self.state_optimizer1.zero_grad()
                loss1.backward()
                state_eval1 = list(self.state_eval1.parameters())
                for i in range(len(state_eval1)):
                    self.state_momentum1[i] = self.state_momentum1[i] * self.beta[0] + state_eval1[i].grad * (
                                1 - self.beta[0])
                    a = state_eval1[i].grad.detach()
                    state_eval1[i].grad = state_eval1[i].grad * self.kp + (
                                self.state_momentum1[i] * self.ki / (1 - self.current_statebeta[0])) + (
                                                  state_eval1[i].grad - self.state_lastgrad1[i]) * self.kd
                    self.state_lastgrad1[i] = a
                self.state_optimizer1.step()
                loss2 = self.loss(y_batch, Q_batch2).cuda()
                self.state_optimizer2.zero_grad()
                loss2.backward()
                state_eval2 = list(self.state_eval2.parameters())
                for i in range(len(state_eval2)):
                    self.state_momentum2[i] = self.state_momentum2[i] * self.beta[0] + state_eval2[i].grad * (
                                1 - self.beta[0])
                    a = state_eval2[i].grad.detach()
                    state_eval2[i].grad = state_eval2[i].grad * self.kp + (
                                self.state_momentum1[i] * self.ki / (1 - self.current_statebeta[0])) + (
                                                  state_eval2[i].grad - self.state_lastgrad2[i]) * self.kd
                    self.state_lastgrad2[i] = a
                self.state_optimizer2.step()
                if self.step_index % self.action_renew_steps == 0:
                    self.current_actionbeta[0] *= self.beta[0]
                    if random.uniform(0, 1) < 0.5:
                        new_action = (self.action_eval(state_batch) * self.max_action).float()
                        Q_val = self.state_eval1(torch.cat([state_batch, new_action], dim=1))
                    else:
                        new_action = (self.action_eval(state_batch) * self.max_action).float()
                        Q_val = self.state_eval2(torch.cat([state_batch, new_action], dim=1))
                    loss3 = (-Q_val.sum() / self.batch_size)
                    self.action_optimizer.zero_grad()
                    loss3.backward()
                    action_eval = list(self.action_eval.parameters())
                    for i in range(len(action_eval)):
                        self.action_momentum[i] = self.action_momentum[i] * self.beta[0] + action_eval[i].grad * (
                                1 - self.beta[0])
                        a = action_eval[i].grad.detach()
                        action_eval[i].grad = action_eval[i].grad * self.kp + (
                                    self.action_momentum[i] * self.ki / (1 - self.current_actionbeta[0])) + (
                                                      action_eval[i].grad - self.action_lastgrad[i]) * self.kd
                        self.action_lastgrad[i] = a
                    self.action_optimizer.step()
            elif self.optim_algorithm == 6:
                '''自适应学习率PID'''
                self.current_statebeta[0] *= self.beta[0]
                self.current_statebeta[1] *= self.beta[1]
                loss1 = self.loss(y_batch, Q_batch1).cuda()
                self.state_optimizer1.zero_grad()
                loss1.backward()
                state_eval1 = list(self.state_eval1.parameters())
                for i in range(len(state_eval1)):
                    self.state_momentum1[i] = self.state_momentum1[i] * self.beta[0] + state_eval1[i].grad * (
                                1 - self.beta[0])
                    a = state_eval1[i].grad.detach()
                    self.state_vt1[i] = self.state_vt1[i] * self.beta[1] + (1 - self.beta[1]) * torch.pow(a, 2)
                    current_vt = torch.pow(self.state_vt1[i] / (1 - self.current_statebeta[1]), 0.5) + self.RMSepsilon
                    state_eval1[i].grad = (state_eval1[i].grad * self.kp + (
                                self.state_momentum1[i] * self.ki / (1 - self.current_statebeta[0])) + (
                                                   state_eval1[i].grad - self.state_lastgrad1[
                                               i]) * self.kd) / current_vt
                    self.state_lastgrad1[i] = a
                self.state_optimizer1.step()
                loss2 = self.loss(y_batch, Q_batch2).cuda()
                self.state_optimizer2.zero_grad()
                loss2.backward()
                state_eval2 = list(self.state_eval2.parameters())
                for i in range(len(state_eval2)):
                    self.state_momentum2[i] = self.state_momentum2[i] * self.beta[0] + state_eval2[i].grad * (
                                1 - self.beta[0])
                    a = state_eval2[i].grad.detach()
                    self.state_vt2[i] = self.state_vt2[i] * self.beta[1] + (1 - self.beta[1]) * torch.pow(a, 2)
                    current_vt = torch.pow(self.state_vt2[i] / (1 - self.current_statebeta[1]), 0.5) + self.RMSepsilon
                    state_eval2[i].grad = (state_eval2[i].grad * self.kp + (
                                self.state_momentum2[i] * self.ki / (1 - self.current_statebeta[0])) + (
                                                   state_eval2[i].grad - self.state_lastgrad2[
                                               i]) * self.kd) / current_vt
                    self.state_lastgrad2[i] = a
                self.state_optimizer2.step()
                if self.step_index % self.action_renew_steps == 0:
                    self.current_actionbeta[0] *= self.beta[0]
                    self.current_actionbeta[1] *= self.beta[1]
                    if random.uniform(0, 1) < 0.5:
                        new_action = (self.action_eval(state_batch) * self.max_action).float()
                        Q_val = self.state_eval1(torch.cat([state_batch, new_action], dim=1))
                    else:
                        new_action = (self.action_eval(state_batch) * self.max_action).float()
                        Q_val = self.state_eval2(torch.cat([state_batch, new_action], dim=1))
                    loss3 = (-Q_val.sum() / self.batch_size)
                    self.action_optimizer.zero_grad()
                    loss3.backward()
                    action_eval = list(self.action_eval.parameters())
                    for i in range(len(action_eval)):
                        self.action_momentum[i] = self.action_momentum[i] * self.beta[0] + action_eval[i].grad * (
                                1 - self.beta[0])
                        a = action_eval[i].grad.detach()
                        self.action_vt[i] = self.action_vt[i] * self.beta[1] + (1 - self.beta[1]) * torch.pow(a, 2)
                        current_vt = torch.pow(self.action_vt[i] / (1 - self.current_actionbeta[1]),
                                               0.5) + self.RMSepsilon
                        action_eval[i].grad = (action_eval[i].grad * self.kp + (
                                    self.action_momentum[i] * self.ki / (1 - self.current_actionbeta[0])) + (
                                                       action_eval[i].grad - self.action_lastgrad[
                                                   i]) * self.kd) / current_vt
                        self.action_lastgrad[i] = a
                    self.action_optimizer.step()
            elif self.optim_algorithm == 7:
                '''自编Adam'''
                self.current_statebeta[0] *= self.beta[0]
                self.current_statebeta[1] *= self.beta[1]
                loss1 = self.loss(y_batch, Q_batch1).cuda()
                self.state_optimizer1.zero_grad()
                loss1.backward()
                state_eval1 = list(self.state_eval1.parameters())
                for i in range(len(state_eval1)):
                    self.state_momentum1[i] = self.state_momentum1[i] * self.beta[0] + state_eval1[i].grad * (
                            1 - self.beta[0])
                    a = state_eval1[i].grad.detach()
                    self.state_vt1[i] = self.state_vt1[i] * self.beta[1] + (1 - self.beta[1]) * torch.pow(a, 2)
                    current_vt = torch.pow(self.state_vt1[i] / (1 - self.current_statebeta[1]), 0.5) + self.RMSepsilon
                    state_eval1[i].grad = (self.state_momentum1[i] / (1 - self.current_statebeta[0])) / current_vt
                self.state_optimizer1.step()
                loss2 = self.loss(y_batch, Q_batch2).cuda()
                self.state_optimizer2.zero_grad()
                loss2.backward()
                state_eval2 = list(self.state_eval2.parameters())
                for i in range(len(state_eval2)):
                    self.state_momentum2[i] = self.state_momentum2[i] * self.beta[0] + state_eval2[i].grad * (
                            1 - self.beta[0])
                    a = state_eval2[i].grad.detach()
                    self.state_vt2[i] = self.state_vt2[i] * self.beta[1] + (1 - self.beta[1]) * torch.pow(a, 2)
                    current_vt = torch.pow(self.state_vt2[i] / (1 - self.current_statebeta[1]), 0.5) + self.RMSepsilon
                    state_eval2[i].grad = (self.state_momentum2[i] / (1 - self.current_statebeta[0])) / current_vt
                self.state_optimizer2.step()
                if self.step_index % self.action_renew_steps == 0:
                    self.current_actionbeta[0] *= self.beta[0]
                    self.current_actionbeta[1] *= self.beta[1]
                    if random.uniform(0, 1) < 0.5:
                        new_action = (self.action_eval(state_batch) * self.max_action).float()
                        Q_val = self.state_eval1(torch.cat([state_batch, new_action], dim=1))
                    else:
                        new_action = (self.action_eval(state_batch) * self.max_action).float()
                        Q_val = self.state_eval2(torch.cat([state_batch, new_action], dim=1))
                    loss3 = (-Q_val.sum() / self.batch_size)
                    self.action_optimizer.zero_grad()
                    loss3.backward()
                    action_eval = list(self.action_eval.parameters())
                    for i in range(len(action_eval)):
                        self.action_momentum[i] = self.action_momentum[i] * self.beta[0] + action_eval[i].grad * (
                                1 - self.beta[0])
                        a = action_eval[i].grad.detach()
                        self.action_vt[i] = self.action_vt[i] * self.beta[1] + (1 - self.beta[1]) * torch.pow(a, 2)
                        current_vt = torch.pow(self.action_vt[i] / (1 - self.current_actionbeta[1]),
                                               0.5) + self.RMSepsilon
                        action_eval[i].grad = (self.action_momentum[i] / (1 - self.current_actionbeta[0])) / current_vt
                    self.action_optimizer.step()
            elif self.optim_algorithm == 8:
                '''自编RMSprop'''
                self.current_statebeta[1] *= self.beta[1]
                loss1 = self.loss(y_batch, Q_batch1).cuda()
                self.state_optimizer1.zero_grad()
                loss1.backward()
                state_eval1 = list(self.state_eval1.parameters())
                for i in range(len(state_eval1)):
                    a = state_eval1[i].grad.detach()
                    self.state_vt1[i] = self.state_vt1[i] * self.beta[1] + (1 - self.beta[1]) * torch.pow(a, 2)
                    current_vt = torch.pow(self.state_vt1[i] / (1 - self.current_statebeta[1]), 0.5) + self.RMSepsilon
                    state_eval1[i].grad = a / current_vt
                self.state_optimizer1.step()
                loss2 = self.loss(y_batch, Q_batch2).cuda()
                self.state_optimizer2.zero_grad()
                loss2.backward()
                state_eval2 = list(self.state_eval2.parameters())
                for i in range(len(state_eval2)):
                    a = state_eval2[i].grad.detach()
                    self.state_vt2[i] = self.state_vt2[i] * self.beta[1] + (1 - self.beta[1]) * torch.pow(a, 2)
                    current_vt = torch.pow(self.state_vt2[i] / (1 - self.current_statebeta[1]), 0.5) + self.RMSepsilon
                    state_eval2[i].grad = a / current_vt
                self.state_optimizer2.step()
                if self.step_index % self.action_renew_steps == 0:
                    self.current_actionbeta[1] *= self.beta[1]
                    if random.uniform(0, 1) < 0.5:
                        new_action = (self.action_eval(state_batch) * self.max_action).float()
                        Q_val = self.state_eval1(torch.cat([state_batch, new_action], dim=1))
                    else:
                        new_action = (self.action_eval(state_batch) * self.max_action).float()
                        Q_val = self.state_eval2(torch.cat([state_batch, new_action], dim=1))
                    loss3 = (-Q_val.sum() / self.batch_size)
                    self.action_optimizer.zero_grad()
                    loss3.backward()
                    action_eval = list(self.action_eval.parameters())
                    for i in range(len(action_eval)):
                        a = action_eval[i].grad.detach()
                        self.action_vt[i] = self.action_vt[i] * self.beta[1] + (1 - self.beta[1]) * torch.pow(a, 2)
                        current_vt = torch.pow(self.action_vt[i] / (1 - self.current_actionbeta[1]),
                                               0.5) + self.RMSepsilon
                        action_eval[i].grad = a / current_vt
                    self.action_optimizer.step()

        else:
            self.replay_memory_store.append([state, action, reward, next_state, done])
