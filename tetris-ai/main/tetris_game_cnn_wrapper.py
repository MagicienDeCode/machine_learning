import math
import gymnasium as gym
import numpy as np
import random
import cv2

from tetris_game import TetrisGame

ROTATE = 0
DOWN = 1
LEFT = 2
RIGHT = 3

class TetrixGameCNNWrapper(gym.Env):
    def __init__(self, seed=42, board_h=20, board_w=10, silent_mode=True, limit_step=True):
        super().__init__()
        self.game = TetrisGame(seed=seed, board_h=board_h, board_w=board_w, silent_mode=silent_mode)
        self.game.reset()
        self.silent_mode = silent_mode
        self.action_space = gym.spaces.Discrete(4)  # ROTATE, DOWN, LEFT, RIGHT

        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(84, 84, 3), dtype=np.uint8)

        self.board_h = board_h
        self.board_w = board_w
        self.done = False

        self.seed_value = seed

        if limit_step:
            self.step_limit = 1000  # Maximum steps per episode
        else:
            self.step_limit = 1000000000
        self.step_counter = 0

        # Track previous score for reward calculation
        self.prev_score = 0
        self.prev_lines_cleared = 0

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
        self.step_counter = 0
        self.prev_score = 0
        self.prev_lines_cleared = 0
        return self._generate_observation(), {}

    def _generate_observation(self):
        """
        Generate a visual observation of the tetris game state.
        Returns an 84x84x3 RGB image.
        """
        # Create observation with hidden area included
        obs = np.zeros((self.board_h + self.game.hidden_h, self.board_w, 3), dtype=np.uint8)

        # Draw placed pieces on the board (light grey)
        for y in range(self.board_h + self.game.hidden_h):
            for x in range(self.board_w):
                if self.game.board[y][x]:
                    obs[y, x] = [180, 180, 180]  # Light grey for placed pieces

        # Draw current falling piece (in its original color)
        if self.game.current_tetrix:
            cells = self.game.current_tetrix.get_cells()
            color = self.game.current_tetrix.color
            for x, y in cells:
                if 0 <= y < self.board_h + self.game.hidden_h and 0 <= x < self.board_w:
                    obs[y, x] = color[::-1]  # Convert RGB to BGR for OpenCV

        # Resize to 84x84 using nearest neighbor interpolation
        obs_resized = cv2.resize(obs, (84, 84), interpolation=cv2.INTER_NEAREST)

        # Convert BGR back to RGB
        obs_resized = cv2.cvtColor(obs_resized, cv2.COLOR_BGR2RGB)

        return obs_resized

    def step(self, action):
        self.done, current_score = self.game.step(action)
        obs = self._generate_observation()

        self.step_counter += 1

        # Calculate reward based on score change and game state
        reward = 0.0

        # Calculate lines cleared this step
        score_diff = current_score - self.prev_score

        # Info dictionary
        info = {
            'score': current_score,
            'level': self.game.level,
            'lines_cleared_this_step': score_diff // (100 * self.game.level) if self.game.level > 0 else 0,
            'step': self.step_counter
        }

        # Check if step limit exceeded
        if self.step_counter > self.step_limit:
            self.done = True

        # Game over penalty
        if self.done:
            if current_score == 0:
                reward = -10.0  # Large penalty for immediate game over
            else:
                reward = -5.0 + (current_score / 1000.0)  # Smaller penalty with score bonus
            if not self.silent_mode and current_score > 0:
                self.game.game_over_sound.play()
        else:
            # Reward for clearing lines
            if score_diff > 0:
                lines_cleared = score_diff // (100 * self.game.level) if self.game.level > 0 else 0
                if lines_cleared == 1:
                    reward = 1.0
                elif lines_cleared == 2:
                    reward = 3.0
                elif lines_cleared == 3:
                    reward = 6.0
                elif lines_cleared >= 4:
                    reward = 10.0  # Tetris bonus
            else:
                # Small negative reward for each step without clearing lines
                reward = -0.01

        self.prev_score = current_score

        return obs, reward, self.done, False, info

    def render(self):
        if not self.silent_mode:
            self.game.render()

    def get_action_mask(self):
        """
        Returns a boolean mask indicating which actions are valid.
        Returns: np.array of shape (4,) with True for valid actions, False for invalid
        """
        mask = np.ones(4, dtype=np.bool_)

        # ROTATE (0) - check if rotation is valid
        mask[ROTATE] = self.game._can_rotate()

        # DOWN (1) - always valid (will place piece if can't move down)
        mask[DOWN] = True

        # LEFT (2) - check if can move left
        mask[LEFT] = self.game._can_move(-1, 0)

        # RIGHT (3) - check if can move right
        mask[RIGHT] = self.game._can_move(1, 0)

        return mask

    def close(self):
        if hasattr(self.game, 'screen') and self.game.screen is not None:
            import pygame
            pygame.quit()
