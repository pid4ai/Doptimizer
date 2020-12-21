import valueSAC_settings as env
import pygame
import numpy as np
import random
import matplotlib.pyplot as plt
import time


def main():
    pygame.init()
    settings = env.Settings()
    screen = pygame.display.set_mode((settings.screen_width, settings.screen_height))
    pygame.display.set_caption(settings.title)
    car = env.Car(screen)

    TEST = settings.TEST
    STEP = settings.STEP
    EPISODE = settings.EPISODE

    matG = settings.matG
    matH = settings.matH
    max_action = settings.max_action

    a1 = []

    agent = env.SAC(4, 1)

    for i in range(EPISODE):
        state = np.zeros(6)

        screen.fill(settings.bg_color)
        car.xshift(state[0])
        car.draw_pendulum(screen, state[0], state[1],state[2])
        pygame.display.flip()

        total_reward = 0

        for step in range(STEP):
            a = agent.policy_net.get_action(np.concatenate([state[1:3], state[4:6]]))
            action = np.tanh(a) * max_action
            next_state = np.dot(matG,state)+matH*(action+random.uniform(-0.05,0.05))
            total_reward += 1
            if abs(next_state[1]) < 0.01 and abs(next_state[2]) < 0.01:
                reward = 1
            elif abs(next_state[1]) < 0.03 and abs(next_state[2]) < 0.03:
                reward = 0.5
            elif abs(next_state[1]) < 0.06 and abs(next_state[2]) < 0.06:
                reward = -0.5
            elif abs(next_state[1]) < 0.1 and abs(next_state[2]) < 0.1:
                reward = -2
            else:
                reward = -5
            screen.fill(settings.bg_color)
            car.xshift(next_state[0])
            car.draw_pendulum(screen, next_state[0], next_state[1],next_state[2])
            pygame.display.flip()

            if abs(next_state[0]) > 2 or abs(next_state[1]) > 0.2 or abs(next_state[2]) > 0.2:
                done = 1
            else:
                done = 0
            agent.SAC_training(np.concatenate([state[1:3], state[4:6]]), a, reward, np.concatenate([next_state[1:3], next_state[4:6]]), done)
            state = next_state
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
                    action = np.tanh(agent.policy_net.get_action(np.concatenate([state[1:3],state[4:6]]))) * max_action
                    state = np.dot(matG,state)+matH*(action+random.uniform(-0.05,0.05))
                    total_reward += 1
                    screen.fill(settings.bg_color)
                    car.xshift(state[0])
                    car.draw_pendulum(screen, state[0], state[1],state[2])
                    pygame.display.flip()
                    time.sleep(0.003)
                    if abs(state[0]) >2 or abs(state[1]) > 0.2 or abs(state[2]) > 0.2:
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
                plt.title('value SAC 2nd pendulum')
                plt.xlabel('episodes')
                plt.ylabel('steps')
                plt.show()
                break



if __name__ == '__main__':
    main()