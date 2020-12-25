import pygame
import pend2nd_TD3_opt_test as inv

import numpy as np
import random
import matplotlib.pyplot as plt
import time


def main():
    pygame.init()
    settings = inv.Settings()
    screen = pygame.display.set_mode((settings.screen_width, settings.screen_height))
    pygame.display.set_caption(settings.title)
    car = inv.Car(screen)

    TEST = settings.TEST
    STEP = settings.STEP
    EPISODE = settings.EPISODE

    matG = settings.matG
    matH = settings.matH

    a1 = []

    agent = inv.DeepQNetwork(3, optim_algorithm=6)

    for i in range(EPISODE):
        state = np.zeros(6)
        noise_state = state + np.random.randn(6) * settings.noise
        screen.fill(settings.bg_color)
        car.xshift(state[0])
        car.draw_pendulum(screen, state[0], state[1], state[2])
        pygame.display.flip()

        total_reward = 0

        for step in range(STEP):
            action = agent.action(noise_state)
            next_state = np.dot(matG,state)+matH*(action + np.random.randn() * 0.001)
            noise_next_state = next_state + np.random.randn(6) * settings.noise
            total_reward += 1
            if abs(noise_next_state[1]) < 0.01 and abs(noise_next_state[2]) < 0.01 and abs(noise_next_state[0]) < 1:
                reward = 1
            elif abs(noise_next_state[1]) < 0.03 and abs(noise_next_state[2]) < 0.03 and abs(noise_next_state[0]) < 1:
                reward = 0.75
            elif abs(noise_next_state[1]) < 0.06 and abs(noise_next_state[2]) < 0.06:
                reward = 0.5
            elif abs(noise_next_state[1]) < 0.03 and abs(noise_next_state[2]) < 0.03:
                reward = 0.25
            else:
                reward = 0
            screen.fill(settings.bg_color)
            car.xshift(next_state[0])
            car.draw_pendulum(screen, next_state[0], next_state[1], next_state[2])
            pygame.display.flip()

            if abs(next_state[0]) > 2 or abs(next_state[1]) > 0.2 or abs(next_state[2]) > 0.2:
                done = 1
            else:
                done = 0
            agent.perceive(noise_state, action, reward, noise_next_state, done)
            agent.model_eval(state, next_state, action)
            state = next_state
            noise_state = noise_next_state
            if done:
                a1.append(total_reward)
                break

        if i % 5 == 0:
            #agent.copy_network()
            total_reward = 0
            for _ in range(TEST):
                state = np.zeros(6)

                screen.fill(settings.bg_color)
                car.xshift(state[0])
                car.draw_pendulum(screen, state[0], state[1], state[2])
                pygame.display.flip()
                time.sleep(0.003)

                for step in range(STEP):
                    action = agent.action(state, explore_sign=False)
                    state = np.dot(matG,state)+matH*(action + np.random.randn() * 0.001)
                    total_reward += 1
                    screen.fill(settings.bg_color)
                    car.xshift(state[0])
                    car.draw_pendulum(screen, state[0], state[1], state[2])
                    pygame.display.flip()
                    time.sleep(0.003)
                    if abs(state[0]) > 2 or abs(state[1]) > 0.2 or abs(state[2]) > 0.2:
                        done = 1
                    else:
                        done = 0
                    if done:
                        break
            ave_reward = total_reward / TEST
            print('episodes:', i, 'average_reward:',ave_reward)
            a1.append(ave_reward)
            if i % 300 == 0 and i > 100:
                plt.plot(agent.modelloss)
                agent.modelloss = []
            if ave_reward >= 5000 or i > EPISODE/2:
                plt.plot(a1)
                if agent.optim_algorithm == 0:
                    plt.title('2nd pendulum TD3 '+'Adam')
                elif agent.optim_algorithm == 1:
                    plt.title('2nd pendulum TD3 '+'RMSprop')
                elif agent.optim_algorithm == 2:
                    plt.title('2nd pendulum TD3 '+'Adagrad')
                elif agent.optim_algorithm == 3:
                    plt.title('2nd pendulum TD3 '+'SGD')
                elif agent.optim_algorithm == 4:
                    plt.title('2nd pendulum TD3 '+'Momentum')
                elif agent.optim_algorithm == 5:
                    plt.title('2nd pendulum TD3 '+'PID ki=0')
                elif agent.optim_algorithm == 6:
                    plt.title('2nd pendulum TD3 '+'RMSpropPID')
                elif agent.optim_algorithm == 7:
                    plt.title('2nd pendulum TD3 '+'Adam1')
                elif agent.optim_algorithm == 8:
                    plt.title('2nd pendulum TD3 '+'RMSprop1')
                plt.xlabel('episodes')
                plt.ylabel('steps')
                plt.show()
                break



if __name__ == '__main__':
    main()