import valueSAC_settings as env
import pygame
import numpy as np
import random
import matplotlib.pyplot as plt
import time

settings = env.Settings()
TEST = settings.TEST
STEP = settings.STEP
EPISODE = settings.EPISODE

matG = settings.matG
matH = settings.matH
max_action = settings.max_action
agent = env.SAC(4, 1)
def main():
    pygame.init()

    '''
    screen = pygame.display.set_mode((settings.screen_width, settings.screen_height))
    pygame.display.set_caption(settings.title)
    car = env.Car(screen)
    '''

    a1 = []

    for i in range(EPISODE):
        state = np.zeros(4)
        '''
        screen.fill(settings.bg_color)
        car.xshift(state[0])
        car.draw_pendulum(screen, state[0], state[1])
        pygame.display.flip()
        '''
        total_reward = 0

        for step in range(STEP):
            a = agent.policy_net.get_action(state)
            action = np.tanh(a) * max_action
            next_state = np.dot(matG,state)+matH*(action+random.uniform(-0.02,0.02))
            total_reward += 1
            if abs(next_state[1]) < 0.01 or abs(state[0]) > 0.5:
                reward = 1
            elif abs(next_state[1]) < 0.03 or abs(state[0]) > 1:
                reward = 0.5
            elif abs(next_state[1]) < 0.06 or abs(state[0]) > 1.5:
                reward = -0.5
            elif abs(next_state[1]) < 0.1 or abs(state[0]) > 2:
                reward = -2
            else:
                reward = -5
            '''
            screen.fill(settings.bg_color)
            car.xshift(next_state[0])
            car.draw_pendulum(screen, next_state[0], next_state[1])
            pygame.display.flip()
            '''
            if abs(next_state[0]) > 2 or abs(next_state[1]) > 0.2:
                done = 1
            else:
                done = 0
            agent.SAC_training(state, a, reward, next_state, done)
            state = next_state
            if done:
                a1.append(total_reward)
                break

        if i % 5 == 0:
            #agent.copy_network()
            total_reward = 0
            for _ in range(TEST):
                state = np.zeros(4)
                '''
                screen.fill(settings.bg_color)
                car.xshift(state[0])
                car.draw_pendulum(screen, state[0], state[1])
                pygame.display.flip()
                time.sleep(0.003)
                '''
                for step in range(STEP):
                    action = np.tanh(agent.policy_net.get_action(state)) * max_action
                    state = np.dot(matG,state)+matH*(action+random.uniform(-0.02, 0.02))
                    total_reward += 1
                    '''
                    screen.fill(settings.bg_color)
                    car.xshift(state[0])
                    car.draw_pendulum(screen, state[0], state[1])
                    pygame.display.flip()
                    time.sleep(0.003)
                    '''
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
            if ave_reward >= STEP or i > EPISODE - 5:
                return(a1)


save_sign = 'RL'
agent.get_task_message()
rewards = []
test_labels = []

while(1):
    agent.set_current_parameters()
    if agent.end_sign == 1:
        for i in range(len(rewards)):
            plt.plot(range(len(rewards[i])), rewards[i])
        plt.legend(test_labels)
        plt.savefig('/home/chen/programs/Doptimizer/data/matplotlib/' + str(save_sign))
        plt.show()
    else:
        agent.task_initialize()
        agent.set_optimizers()
        rewards.append(main())
        test_labels.append(agent.algorithm_labels[agent.current_algorithm] + ',' + str(agent.current_lr) + ',' + str(agent.current_beta))
        if agent.derivative_sign[agent.current_algorithm] == 1:
            test_labels[-1] = test_labels[-1] + ',' + str(agent.current_PIparameter)
