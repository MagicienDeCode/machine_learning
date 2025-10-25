import time
import random

import torch
import pygame

from sb3_contrib import MaskablePPO

from tetris_game_cnn_wrapper import TetrixGameCNNWrapper

if torch.backends.mps.is_available():
    MODEL_PATH = r"trained_models_cnn_mps/ppo_tetris_final"
else:
    MODEL_PATH = r"trained_models_cnn/ppo_tetris_final"

NUM_EPISODES = 10
RENDER = True
FRAME_DELAY = 0.05  # 0.01 fast, 0.05 slow
ROUND_DELAY = 5

seed = random.randint(0, 10000000000)

if RENDER:
    env = TetrixGameCNNWrapper(seed=seed, silent_mode=False, limit_step=False)
else:
    env = TetrixGameCNNWrapper(seed=seed, silent_mode=True, limit_step=False)

# Load the trained model
model = MaskablePPO.load(MODEL_PATH)

total_rewards = 0
total_score = 0
min_score = 1000000000
max_score = 0

for episode in range(NUM_EPISODES):
    # Gymnasium Env.reset() returns (obs, info). Ensure we only pass the observation to the model.
    reset_ret = env.reset()
    if isinstance(reset_ret, tuple) and len(reset_ret) == 2:
        obs, reset_info = reset_ret
    else:
        obs = reset_ret
    episode_reward = 0
    done = False
    step = 0
    info = None
    sum_step_reward = 0
    total_lines_cleared = 0

    print(f"==========Episode {episode+1}/{NUM_EPISODES} ==========")
    while not done:
        # model.predict expects a raw observation (np.array or dict), not a (obs, info) tuple.
        action, _ = model.predict(obs, action_masks=env.get_action_mask())
        prev_mask = env.get_action_mask()
        step += 1
        # Gymnasium Env.step() can return either 4-tuple (obs, reward, done, info)
        # or 5-tuple (obs, reward, terminated, truncated, info). Handle both.
        step_ret = env.step(action)
        if isinstance(step_ret, tuple) and len(step_ret) == 5:
            obs, reward, terminated, truncated, info = step_ret
            done = terminated or truncated
        else:
            obs, reward, done, info = step_ret

        episode_reward += reward

        if done:
            last_action = ['ROTATE', 'DOWN', 'LEFT', 'RIGHT'][action]
            print(f"Game over! Last action: {last_action}, final reward: {reward:.4f}")
        elif info['lines_cleared_this_step'] > 0:
            lines_cleared = info['lines_cleared_this_step']
            total_lines_cleared += lines_cleared
            print(f"Lines cleared: {lines_cleared} at step {step}!, line reward: {reward:.4f}, level: {info['level']}")
            sum_step_reward = 0
        else:
            sum_step_reward += reward

        if RENDER:
            pygame.event.pump()  # Process pygame events to keep window responsive
            env.render()
            time.sleep(FRAME_DELAY)

    episode_score = info['score']
    min_score = min(min_score, episode_score)
    max_score = max(max_score, episode_score)

    print(f"Episode {episode+1} ended. Score: {episode_score}, Lines cleared: {total_lines_cleared}, Level: {info['level']}, Steps: {step}, Total reward: {episode_reward:.2f}")
    total_rewards += episode_reward
    total_score += episode_score
    if RENDER:
        # Process events during delay to keep window responsive
        end_time = time.time() + ROUND_DELAY
        while time.time() < end_time:
            pygame.event.pump()
            time.sleep(0.1)

env.close()
print(f"========== Summary ==========")
print(f"Average score over {NUM_EPISODES} episodes: {total_score/NUM_EPISODES:.2f}, Min score: {min_score}, Max score: {max_score}")
print(f"Average reward over {NUM_EPISODES} episodes: {total_rewards/NUM_EPISODES:.2f}")
