import time
import matplotlib.pyplot as plt
from dqn import DQN
from grid_world import GridWorldEnv




EPISODE = 300
STEP = 100
if __name__ == "__main__":
    env = GridWorldEnv()
    dqn = DQN(env)
    step_list = []

    for episode in range(EPISODE):
        state = env.reset()
        for step in range(STEP):
            env.render()
            action = dqn.egreedy_action(state)
            next_state, reward, done, _ = env.step(action)
            # print(action)
            # print(next_state)
            # print(reward)
            # print(done)
            # print("="*20)
            dqn.perceive(state, action, reward, next_state, done)
            state = next_state
            if done:
                print("finished! at step: {} at episode: {}".format(step, episode))
                step_list.append(step)
                break
            
            if step == STEP -1:
                print("Unfinihsed at step: {} at episode: {}".format(step, episode))
                step_list.append(step)
                break


    plt.figure()
    plt.plot(step_list)
    plt.show()

