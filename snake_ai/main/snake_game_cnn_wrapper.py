import math
import gymnasium as gym
import numpy as np
import random
import cv2

from snake_game import SnakeGame

UP = 0
DOWN = 1
LEFT = 2
RIGHT = 3

class SnakeGameCNNWrapper(gym.Env):
    def __init__(self, seed=42, board_size=12, silent_mode=True, limit_step=True):
        super().__init__()
        self.game = SnakeGame(seed=seed, board_size=board_size, silent_mode=silent_mode)
        self.game.reset()
        self.silent_mode = silent_mode
        self.action_space = gym.spaces.Discrete(4)  # UP, DOWN, LEFT, RIGHT
        
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(84, 84, 3), dtype=np.uint8)

        self.board_size = board_size
        self.grid_size = board_size ** 2
        self.init_snake_size = len(self.game.snake)
        self.max_growth = self.grid_size - self.init_snake_size
        self.done = False

        self.seed = seed

        if limit_step:
            self.step_limit = 4 * self.grid_size
        else:
            self.step_limit = 1000000000
        self.reward_step_counter = 0

    def seed(self, seed=None):
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        random.seed(seed)
        np.random.seed(seed)
        return seed
    
    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self.seed(seed)
        self.game.reset()
        self.done = False
        self.reward_step_counter = 0
        return self._generate_observation(), {}
    
    def _generate_observation(self):
        obs = np.zeros((self.board_size, self.board_size), dtype=np.uint8)

        # set snake body gray
        if len(self.game.snake) > 0:
            rows = np.array([r for r,c in self.game.snake])
            cols = np.array([c for r,c in self.game.snake])
            obs[rows, cols] = np.linspace(200,50,len(self.game.snake), dtype=np.uint8)

        # stack into 3-channel
        obs = np.stack((obs, obs, obs), axis=-1)

        # draw food (red)
        r, c = self.game.food
        obs[r, c] = [0,0,255]

        # draw head (green) and tail (blue)
        r, c = self.game.snake[0]
        obs[r, c] = [0,255,0]
        r, c = self.game.snake[-1]
        obs[r, c] = [255,0,0]

        # resize to 84x84
        obs_resized = cv2.resize(obs, (84,84), interpolation=cv2.INTER_NEAREST)
        return obs_resized
    
    def step(self, action):
        self.done, info = self.game.step(action)
        obs = self._generate_observation()

        reward = 0.0
        self.reward_step_counter += 1

        if info['snake_size'] == self.grid_size:
            reward = self.max_growth * 0.1
            self.done = True
            if not self.silent_mode:
                self.game.victory_sound.play()
            return obs, reward, self.done, self.done, info
        
        if self.reward_step_counter > self.step_limit:
            self.done = True
            self.reward_step_counter = 0
        
        if self.done:
            reward = -math.pow(self.max_growth, (self.grid_size - info['snake_size']) / self.max_growth)
            reward = reward * 0.1
            return obs, reward, self.done, self.done, info
        elif info['food_eated']:
            reward = info['snake_size'] / self.grid_size
            self.reward_step_counter = 0
        else:
            # give a tiny reward/penalty to the agent based on whether it is heading towards to the food or not
            # not competing with game over penalty or food reward
            if np.linalg.norm(info['snake_head_position'] - info['food_position']) < np.linalg.norm(info['prev_snake_head_position'] - info['food_position']):
                reward = 1 / info['snake_size']
            else:
                reward = -1 / info['snake_size']
            reward = reward * 0.1
        return obs, reward, self.done, self.done, info
    
    def reder(self):
        self.game.render()

    def get_action_mask(self):
        return np.array([[self._check_action_valid(a) for a in range(self.action_space.n)]])
        
    def _check_action_valid(self, action):
        current_direction = self.game.direction
        snake = self.game.snake
        row,col = snake[0]
        if action == UP:
            if current_direction == DOWN:
                return False
            else:
                row -= 1
        elif action == DOWN:
            if current_direction == UP:
                return False
            else:
                row += 1
        elif action == LEFT:
            if current_direction == RIGHT:
                return False
            else:
                col -= 1
        elif action == RIGHT:
            if current_direction == LEFT:
                return False
            else:
                col += 1
        
        if (row,col) == self.game.food:
            # normlly eat food, snake body will grow, no need to check, but here for code robustness we still check
            game_over = (
                (row < 0 or row >= self.board_size or col < 0 or col >= self.board_size) or
                ((row, col) in snake)
            )
        else:
            # if not eat food, snake tail will move forward, so we can ignore the last segment of the snake body
            game_over = (
                (row < 0 or row >= self.board_size or col < 0 or col >= self.board_size) or
                ((row, col) in snake[:-1])
            )
        return not game_over