import numpy as np
import random
import pygame
import math
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import time
import gym


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


class REINFORCE(nn.Module):

    def __init__(self, lr_descent_mode = 0):
        super(REINFORCE,self).__init__()
        '''超参数设定及网络初始化'''
        self.step_index = 0
        self.lr = 0.01
        self.lr_descent_mode = lr_descent_mode
        self.gamma = 0.97
        self.softmax_actions = [-1, 1]
        self.action_dim = len(self.softmax_actions)
        self.softmax_REINFORCE = nn.Sequential(nn.Linear(4, 10),
                                               nn.ReLU(),
                                               nn.Linear(10, 10),
                                               nn.ReLU(),
                                               nn.Linear(10, self.action_dim),
                                               nn.Softmax()).cuda()

        self.old_net = nn.Sequential(nn.Linear(4, 10),
                                               nn.ReLU(),
                                               nn.Linear(10, 10),
                                               nn.ReLU(),
                                               nn.Linear(10, self.action_dim),
                                               nn.Softmax()).cuda()

        parameters = list(self.softmax_REINFORCE.parameters())
        old_parameters = list(self.old_net.parameters())
        for i in range(len(parameters)):
            old_parameters[i] = parameters[i].detach()
        #self.old_net.training = False
        self.Gaussian_REINFORCE = nn.Sequential(nn.Linear(4, 10),
                                               nn.ReLU(),
                                               nn.Linear(10, 20),
                                               nn.ReLU(),
                                               nn.Linear(10, 1),
                                               nn.Tanh()).cuda()

        self.optimizer = torch.optim.Adam(self.softmax_REINFORCE.parameters(), lr=self.lr)
        self.old_optimizer = torch.optim.Adam(self.old_net.parameters(), lr=self.lr)
        self.episode_count = 0
        self.Gaussian_var = 0.1
        self.beta = 0.9

    def Gaussian_explore_action(self, state):
        state = torch.from_numpy(np.array(state)).cuda().float()
        average = self.Gaussian_REINFORCE(state).detach().cpu().numpy()
        action = np.random.randn() * self.Gaussian_var + average
        return action

    def Gaussian_greedy_action(self, state):
        state = torch.from_numpy(np.array(state)).cuda().float()
        action = self.Gaussian_REINFORCE(state).detach().cpu().numpy()
        return action


    def softmax_explore_action(self, state):
        state = torch.from_numpy(np.array(state)).cuda().float()
        prob = self.softmax_REINFORCE(state).detach().cpu().numpy()
        a = random.uniform(0, 1)
        for i in range(self.action_dim):
            if a <= np.sum(prob[:i+1]):
                action = i
                break
        return action

    def softmax_greedy_action(self, state):
        state = torch.from_numpy(np.array(state)).cuda().float()
        prob = self.softmax_REINFORCE(state).detach().cpu().numpy()
        return np.argmax(prob)

    def softmax_training(self, episode_data):
        '''episode_data标准格式：[state, action, reward, prob]'''
        rewards = np.array([i[2] for i in episode_data])
        actions = torch.from_numpy(np.array([i[1] for i in episode_data])[:, np.newaxis]).cuda()
        states = torch.from_numpy(np.array([i[0] for i in episode_data])).cuda().float()
        discounted_returns = np.zeros(len(episode_data))
        current_return = 0
        for i in reversed(range(len(episode_data))):
            current_return = current_return * self.gamma + rewards[i]
            discounted_returns[i] = current_return
        discounted_returns -= np.average(discounted_returns)
        discounted_returns /= np.std(discounted_returns)
        discounted_returns = torch.from_numpy(discounted_returns).cuda().float()
        prob = torch.sum(self.softmax_REINFORCE(states) * torch.zeros(
            len(episode_data), self.action_dim).cuda().scatter_(1, actions.long(), 1), dim=1)
        log_prob = torch.log(prob)
        loss = -torch.mean(log_prob * discounted_returns)
        self.optimizer.zero_grad()
        loss.backward()
        if self.episode_count == 0:
            self.old_grad = []
            parameters = list(self.softmax_REINFORCE.parameters())
            for i in range(len(parameters)):
                self.old_grad.append(parameters[i].grad.detach())
        else:
            last_prob = torch.sum(self.old_net(states) * torch.zeros(
                len(episode_data), self.action_dim).cuda().scatter_(1, actions.long(), 1), dim=1)
            omega = torch.prod(last_prob / prob)
            print(omega)
            log_last_prob = torch.log(last_prob)
            last_loss = -torch.mean(log_last_prob * discounted_returns)
            last_loss.backward()
            last_parameters = list(self.old_net.parameters())
            new_parameters = list(self.softmax_REINFORCE.parameters())
            for i in range(len(new_parameters)):
                new_grad = new_parameters[i].grad.detach()
                last_grad = last_parameters[i].grad.detach()
                old_grad = self.old_grad[i].detach()
                self.old_grad[i] = (new_grad + self.beta * (old_grad - omega * last_grad)).detach()
                new_parameters[i].grad = self.old_grad[i]
                last_parameters[i] = new_parameters[i].detach()
        self.old_optimizer.zero_grad()
        self.episode_count += 1
        self.optimizer.step()




    def Gaussian_training(self, episode_data):
        '''episode_data标准格式：[state, action, reward, prob]'''
        rewards = np.array([i[2] for i in episode_data])
        actions = torch.from_numpy(np.array([i[1] for i in episode_data])[:, np.newaxis]).cuda()
        states = torch.from_numpy(np.array([i[0] for i in episode_data])).cuda().float()
        discounted_returns = np.zeros(len(episode_data))
        current_return = 0
        for i in reversed(range(len(episode_data))):
            current_return = current_return * self.gamma + rewards[i]
            discounted_returns[i] = current_return
        discounted_returns -= np.average(discounted_returns)
        discounted_returns /= np.std(discounted_returns)
        discounted_returns = torch.from_numpy(discounted_returns).cuda().float()
        log_prob = torch.sum(-(actions - self.Gaussian_REINFORCE(states)) ** 2 / (2 * self.Gaussian_var))
        loss = -torch.mean(log_prob * discounted_returns)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

def Softmax_main():
    pygame.init()
    settings = Settings()
    screen = pygame.display.set_mode((settings.screen_width, settings.screen_height))
    pygame.display.set_caption(settings.title)
    car = Car(screen)

    TEST = settings.TEST
    STEP = settings.STEP
    EPISODE = settings.EPISODE

    matG = settings.matG
    matH = settings.matH

    a1 = []

    agent = REINFORCE(2)

    for i in range(EPISODE):
        state = np.zeros(4)
        screen.fill(settings.bg_color)
        car.xshift(state[0])
        car.draw_pendulum(screen, state[0], state[1])
        pygame.display.flip()

        total_reward = 0
        episode_data = []
        for step in range(STEP):
            action_index = agent.softmax_explore_action(state)
            action = agent.softmax_actions[action_index]
            next_state = np.dot(matG, state)+matH*(action+random.uniform(-0.03,0.03))
            total_reward += 1
            if abs(next_state[1]) < 0.01 and abs(next_state[0]) < 1:
                reward = 1
            elif abs(next_state[1]) < 0.03 and abs(next_state[0]) < 1:
                reward = 0.5
            elif abs(next_state[1]) < 0.06 and abs(next_state[0]) < 2:
                reward = 0
            elif abs(next_state[1]) < 0.1 and abs(next_state[0]) < 2:
                reward = -0.5
            elif abs(next_state[1]) < 0.2 and abs(next_state[0]) < 2:
                reward = -1
            else:
                reward = -5
            screen.fill(settings.bg_color)
            car.xshift(next_state[0])
            time.sleep(0.003)
            car.draw_pendulum(screen, next_state[0], next_state[1])
            pygame.display.flip()

            if abs(next_state[0]) > 2 or abs(next_state[1]) > 0.2:
                done = 1
            else:
                done = 0
            episode_data.append([state, action_index, reward])
            state = next_state

            if done:
                a1.append(total_reward)
                agent.softmax_training(episode_data)
                break

        if i % 5 == 0:
            #agent.copy_network()
            total_reward = 0
            for _ in range(TEST):
                state = np.zeros(4)

                screen.fill(settings.bg_color)
                car.xshift(state[0])
                car.draw_pendulum(screen, state[0], state[1])
                pygame.display.flip()
                time.sleep(0.003)

                for step in range(STEP):
                    action = agent.softmax_actions[agent.softmax_greedy_action(state)]
                    state = np.dot(matG,state)+matH*(action+random.uniform(-0.05,0.05))
                    total_reward += 1
                    screen.fill(settings.bg_color)
                    car.xshift(state[0])
                    car.draw_pendulum(screen, state[0], state[1])
                    pygame.display.flip()
                    time.sleep(0.003)
                    if abs(state[0]) > 2 or abs(state[1]) > 0.2:
                        done = 1
                    else:
                        done = 0
                    #agent.perceive(state,np.array(action), reward,np.array(next_state), done)
                    if done:
                        break
            ave_reward = total_reward / TEST
            print('episodes:', i, 'average_reward:',ave_reward)
            a1.append(ave_reward)
            if ave_reward >= 1000 or i>2000:
                plt.plot(a1)
                plt.xlabel('episodes')
                plt.ylabel('steps')
                plt.show()
                break

def Gaussian_main():
    pygame.init()
    settings = Settings()
    screen = pygame.display.set_mode((settings.screen_width, settings.screen_height))
    pygame.display.set_caption(settings.title)
    car = Car(screen)

    TEST = settings.TEST
    STEP = settings.STEP
    EPISODE = settings.EPISODE

    matG = settings.matG
    matH = settings.matH

    a1 = []

    agent = REINFORCE(2)

    for i in range(EPISODE):
        state = np.zeros(4)
        screen.fill(settings.bg_color)
        car.xshift(state[0])
        car.draw_pendulum(screen, state[0], state[1])
        pygame.display.flip()

        total_reward = 0
        episode_data = []
        for step in range(STEP):
            action = agent.Gaussian_explore_action(state)
            next_state = np.dot(matG, state)+matH*(action+random.uniform(-0.03,0.03))
            total_reward += 1
            if abs(next_state[1]) < 0.01 and abs(next_state[0]) < 1:
                reward = 1
            elif abs(next_state[1]) < 0.03 and abs(next_state[0]) < 1:
                reward = 0.5
            elif abs(next_state[1]) < 0.06 and abs(next_state[0]) < 2:
                reward = 0
            elif abs(next_state[1]) < 0.1 and abs(next_state[0]) < 2:
                reward = -0.5
            elif abs(next_state[1]) < 0.2 and abs(next_state[0]) < 2:
                reward = -1
            else:
                reward = -5
            screen.fill(settings.bg_color)
            car.xshift(next_state[0])
            time.sleep(0.003)
            car.draw_pendulum(screen, next_state[0], next_state[1])
            pygame.display.flip()

            if abs(next_state[0]) > 2 or abs(next_state[1]) > 0.2:
                done = 1
            else:
                done = 0
            episode_data.append([state, action, reward])
            state = next_state

            if done:
                a1.append(total_reward)
                agent.Gaussian_training(episode_data)
                break

        if i % 5 == 0:
            #agent.copy_network()
            total_reward = 0
            for _ in range(TEST):
                state = np.zeros(4)

                screen.fill(settings.bg_color)
                car.xshift(state[0])
                car.draw_pendulum(screen, state[0], state[1])
                pygame.display.flip()
                time.sleep(0.003)

                for step in range(STEP):
                    action = agent.Gaussian_greedy_action(state)
                    state = np.dot(matG,state)+matH*(action+random.uniform(-0.03,0.03))
                    total_reward += 1
                    screen.fill(settings.bg_color)
                    car.xshift(state[0])
                    car.draw_pendulum(screen, state[0], state[1])
                    pygame.display.flip()
                    time.sleep(0.003)
                    if abs(state[0]) > 2 or abs(state[1]) > 0.2:
                        done = 1
                    else:
                        done = 0
                    #agent.perceive(state,np.array(action), reward,np.array(next_state), done)
                    if done:
                        break
            ave_reward = total_reward / TEST
            print('episodes:', i, 'average_reward:',ave_reward)
            a1.append(ave_reward)
            if ave_reward >= 1000 or i>2000:
                plt.plot(a1)
                plt.xlabel('episodes')
                plt.ylabel('steps')
                plt.show()
                break



if __name__ == '__main__':
    Softmax_main()












