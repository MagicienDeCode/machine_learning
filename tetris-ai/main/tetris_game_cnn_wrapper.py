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
    def __init__(self, seed=42, board_h=20, board_w=10, silent_mode=True, limit_step=True, training_phase='early'):
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

        # Track holes for penalty calculation
        self.prev_hole_count = 0

        # Track landing positions
        self.prev_max_height = 0

        # Training phase for weight adjustment
        self.training_phase = training_phase  # 'early' or 'late'

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
        self.prev_hole_count = 0
        self.prev_max_height = 0
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

    def _count_holes(self):
        """
        Count the number of holes in the board.
        A hole is an empty cell with at least one filled cell above it in the same column.
        """
        holes = 0
        for x in range(self.board_w):
            found_block = False
            for y in range(self.board_h + self.game.hidden_h):
                if self.game.board[y][x]:
                    found_block = True
                elif found_block and not self.game.board[y][x]:
                    holes += 1
        return holes

    def _get_max_height(self):
        """
        Get the maximum height of the board (highest filled cell).
        Returns the row index from the bottom (0 = bottom row).
        """
        for y in range(self.board_h + self.game.hidden_h):
            for x in range(self.board_w):
                if self.game.board[y][x]:
                    # Return height from bottom
                    return (self.board_h + self.game.hidden_h) - y
        return 0

    def _get_landing_height(self):
        """
        Calculate the landing height of the last placed piece.
        This is approximated by the change in max height after the piece lands.
        """
        current_max_height = self._get_max_height()
        landing_height = current_max_height - self.prev_max_height
        return max(landing_height, 0)  # Ensure non-negative

    def step(self, action):
        # Store state before action
        prev_holes = self.prev_hole_count
        prev_max_height = self.prev_max_height

        self.done, current_score = self.game.step(action)
        obs = self._generate_observation()

        self.step_counter += 1

        # Calculate state metrics after action
        current_holes = self._count_holes()
        current_max_height = self._get_max_height()

        # Calculate reward based on the new strategy
        reward = 0.0

        # Calculate lines cleared this step
        score_diff = current_score - self.prev_score
        lines_cleared = 0
        if score_diff > 0 and self.game.level > 0:
            lines_cleared = score_diff // (100 * self.game.level)
        elif score_diff > 0:
            lines_cleared = score_diff // 100

        # Info dictionary
        info = {
            'score': current_score,
            'level': self.game.level,
            'lines_cleared_this_step': lines_cleared,
            'step': self.step_counter,
            'holes': current_holes,
            'max_height': current_max_height
        }

        # Check if step limit exceeded
        if self.step_counter > self.step_limit:
            self.done = True

        # Calculate rewards based on Dense Reward A+ strategy
        if self.done:
            # 6. Game over penalty
            reward -= 10.0
            if not self.silent_mode and current_score > 0:
                self.game.game_over_sound.play()
        else:
            # 2. Survival reward: +0.01 for each step survived
            reward += 0.01

            # 1. Line clear rewards
            if lines_cleared > 0:
                # 1.1 Reward +1.0 for each line cleared
                reward += 1.0 * lines_cleared

                # 1.2 Additional +5.0 bonus for clearing 4 lines at once (Tetris)
                if lines_cleared >= 4:
                    reward += 5.0

            # 3. Landing height penalty (when piece lands)
            # A piece lands when the board state changes (holes or height changes)
            if current_max_height != prev_max_height or current_holes != prev_holes:
                landing_height = current_max_height
                reward -= 0.1 * landing_height

            # 4. Hole penalty
            holes_added = current_holes - prev_holes
            if holes_added > 0:
                reward -= 0.5 * holes_added

            # 5. Drop action reward (conditional)
            if action == DOWN:
                # Simple heuristic: reward drop if it leads to good outcomes
                drop_reward = 0.0

                # Condition: If drop clears lines
                if lines_cleared > 0:
                    drop_reward = 0.1
                # Condition: If drop doesn't create new holes and height increase is small
                elif holes_added == 0 and (current_max_height - prev_max_height) <= 1:
                    drop_reward = 0.05
                # Condition: If current height is low (< 10 cells)
                elif current_max_height < 10:
                    drop_reward = 0.05

                # 7. Weight adjustment by training phase
                if self.training_phase == 'late':
                    drop_reward *= 2.0  # Double the reward in late phase

                reward += drop_reward

        # Update tracking variables for next step
        self.prev_score = current_score
        self.prev_hole_count = current_holes
        self.prev_max_height = current_max_height

        return obs, reward, self.done, False, info

    def render(self):
        if not self.silent_mode:
            self.game.render()

    def set_training_phase(self, phase):
        """
        Update the training phase for weight adjustment.
        Args:
            phase (str): 'early' for exploration phase, 'late' for stable phase
        """
        if phase not in ['early', 'late']:
            raise ValueError("Phase must be 'early' or 'late'")
        self.training_phase = phase

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
