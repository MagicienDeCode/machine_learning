import time
import random

import torch

from sb3_contrib import MaskablePPO

from snake_game_cnn_wrapper import SnakeGameCNNWrapper

if torch.backends.mps.is_available():
    MODEL_PATH =  r"trained_models_cnn_mps/ppo_snake_final"
else:
    MODEL_PATH =  r"trained_models_cnn/ppo_snake_final"

NUM_EPISODES = 10
RENDER = True
FRAME_DELAY = 0.05 # 0.01 fast, 0.05 slow
ROUND_DELAY = 5

seed = random.randint(0, 10000000000)

if RENDER:
    env = SnakeGameCNNWrapper(seed=seed, silent_mode=False, limit_step=False)
else:
    env = SnakeGameCNNWrapper(seed=seed, silent_mode=True, limit_step=False)

# Load the trainded model

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

    retry_limit = 9
    print(f"==========Episode {episode+1}/{NUM_EPISODES} ==========")
    while not done:
        # model.predict expects a raw observation (np.array or dict), not a (obs, info) tuple.
        action, _ = model.predict(obs, action_masks=env.get_action_mask())
        prev_mask = env.get_action_mask()
        prev_direction = env.game.direction
        step += 1
        # Gymnasium Env.step() can return either 4-tuple (obs, reward, done, info)
        # or 5-tuple (obs, reward, terminated, truncated, info). Handle both.
        step_ret = env.step(action)
        if isinstance(step_ret, tuple) and len(step_ret) == 5:
            obs, reward, terminated, truncated, info = step_ret
            done = terminated or truncated
        else:
            obs, reward, done, info = step_ret

        if done:
            if info['snake_size'] == env.game.grid_size:
                print(f"Victory! reward: {reward}")
            else:
                last_action = ['UP', 'DOWN', 'LEFT', 'RIGHT'][action]
        elif info['food_eated']:
            print(f"Food eated at step {step}!, food reward: {reward}, step reward: {sum_step_reward}")
        else:
            sum_step_reward += reward
        
        if RENDER:
            env.render()
            time.sleep(FRAME_DELAY)
    
    episode_score = env.game.score
    min_score = min(min_score, episode_score)
    max_score = max(max_score, episode_score)

    snake_size = info['snake_size'] + 1
    print(f"Episode {episode+1} ended. score: {episode_score}, snake size: {snake_size}, steps: {step}, total reward: {info['total_rewards']}")
    total_rewards += episode_reward
    total_score += episode_score
    if RENDER:
        time.sleep(ROUND_DELAY)

env.close()
print(f"========== Summary ==========")
print(f"Average score over {NUM_EPISODES} episodes: {total_score/NUM_EPISODES}, min score: {min_score}, max score: {max_score}")