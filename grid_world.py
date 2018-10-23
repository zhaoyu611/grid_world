import numpy as np
import gym
from gym import spaces
from gym.utils import seeding
from dqn import DQN

class GridWorldEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 50
    }

    def __init__(self):
        self.action_space = spaces.Discrete(4)
        low = np.array([0, 0])
        high = np.array([100, 100])
        self.observation_space = spaces.Box(low, high, dtype=np.float32)

        self.target = np.array([50, 50])
        self.seed()
        self.viewer = None
        self.state = None

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        assert self.action_space.contains(
            action), "action: {0}, type: {1} is invalid".format(action, type(action))

        if action == 0:
            self.state[0] += 25
        elif action == 1:
            self.state[0] -= 25
        elif action == 2:
            self.state[1] += 25
        elif action == 3:
            self.state[1] -= 25

        self.state = np.clip(self.state, 0, 75)
        if (self.state == self.target).all():
            reward = 1
            done = True
        else:
            reward = 0
            done = False

        return self.state, reward, done, {}


    def reset(self):
        self.state = np.array([0, 0])
        return self.state

    def render(self, mode='human'):
        screen_width = 400
        screen_height = 400
        scale = screen_height / 100

        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(screen_width, screen_height)

            line11 = rendering.Line((0, 25 * scale), (100 * scale, 25 * scale))
            self.viewer.add_geom(line11)
            line12 = rendering.Line((0, 50 * scale), (100 * scale, 50 * scale))
            self.viewer.add_geom(line12)
            line13 = rendering.Line((0, 75 * scale), (100 * scale, 75 * scale))
            self.viewer.add_geom(line13)
            line21 = rendering.Line((25 * scale, 0), (25 * scale, 100 * scale))
            self.viewer.add_geom(line21)
            line22 = rendering.Line((50 * scale, 0), (50 * scale, 100 * scale))
            self.viewer.add_geom(line22)
            line23 = rendering.Line((75 * scale, 0), (75 * scale, 100 * scale))
            self.viewer.add_geom(line23)

            draw_target = rendering.make_circle(25 / 2 * scale)
            target_trans = rendering.Transform(translation=(
                (self.target[0] + 25 / 2) * scale, (self.target[0] + 25 / 2) * scale))
            draw_target.add_attr(target_trans)
            draw_target.set_color(255, 255, 0)
            self.viewer.add_geom(draw_target)

            draw_agent = rendering.make_circle(25 / 2 * scale)
            self.agent_trans = rendering.Transform(translation=(
                (self.state[0] + 25 / 2) * scale, (self.state[1] + 25 / 2) * scale))
            draw_agent.add_attr(self.agent_trans)
            draw_agent.set_color(255, 0, 0)
            self.viewer.add_geom(draw_agent)

        if self.viewer is None:
            return None

        self.agent_trans.set_translation((self.state[0] + 25 / 2) * scale, (self.state[1] + 25 / 2) * scale)
        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

    def choose_action(self):
        return self.action_space.sample()

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None
