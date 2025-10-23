import os
import sys
import random
import glob
import re

import torch
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback

from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker

from tetris_game_cnn_wrapper import TetrixGameCNNWrapper

if torch.backends.mps.is_available():
    NUM_ENV = 32*2
else:
    NUM_ENV = 32
LOG_DIR = "logs"

os.makedirs(LOG_DIR, exist_ok=True)

def find_latest_checkpoint(save_dir, name_prefix="ppo_tetris"):
    """
    Find the latest checkpoint in the save directory.
    Returns: (checkpoint_path, steps) or (None, 0) if no checkpoint found
    """
    if not os.path.exists(save_dir):
        return None, 0

    # Look for checkpoint files
    checkpoint_pattern = os.path.join(save_dir, f"{name_prefix}_*_steps.zip")
    checkpoints = glob.glob(checkpoint_pattern)

    if not checkpoints:
        # Check if final model exists
        final_model = os.path.join(save_dir, f"{name_prefix}_final.zip")
        if os.path.exists(final_model):
            return final_model, -1  # -1 indicates final model
        return None, 0

    # Extract step numbers from checkpoint filenames
    checkpoint_steps = []
    for cp in checkpoints:
        match = re.search(r'_(\d+)_steps\.zip$', cp)
        if match:
            steps = int(match.group(1))
            checkpoint_steps.append((cp, steps))

    if not checkpoint_steps:
        return None, 0

    # Return the checkpoint with the highest step count
    latest_checkpoint = max(checkpoint_steps, key=lambda x: x[1])
    return latest_checkpoint

def linear_schedule(initial_value, final_value=0.0):
    if isinstance(initial_value, str):
        initial_value = float(initial_value)
        final_value = float(final_value)
        assert (initial_value > 0.0)
    def scheduler(progress):
        return final_value + (initial_value - final_value) * progress
    return scheduler

def make_env(seed=42):
    def _init():
        env = TetrixGameCNNWrapper(seed=seed)
        env = ActionMasker(env, TetrixGameCNNWrapper.get_action_mask)
        env = Monitor(env)
        return env
    return _init

def main():
    seed_set = set()
    while len(seed_set) < NUM_ENV:
        seed_set.add(random.randint(0, 1000000000))
    env = SubprocVecEnv([make_env(seed) for seed in seed_set])

    # set the save directory
    if torch.backends.mps.is_available():
        save_dir = "trained_models_cnn_mps"
        device = "mps"
    else:
        save_dir = "trained_models_cnn"
        device = "cuda"
    os.makedirs(save_dir, exist_ok=True)

    # Check for existing checkpoint
    checkpoint_path, checkpoint_steps = find_latest_checkpoint(save_dir)

    if checkpoint_path:
        print(f"==========================================================")
        print(f"Found existing checkpoint: {checkpoint_path}")
        if checkpoint_steps == -1:
            print(f"Loading final model (training was completed)")
        else:
            print(f"Resuming training from step {checkpoint_steps}")
        print(f"==========================================================")

        # Load the existing model
        model = MaskablePPO.load(checkpoint_path, env=env, device=device)
        print(f"Model loaded successfully!")
    else:
        print(f"==========================================================")
        print(f"No existing checkpoint found. Starting training from scratch.")
        print(f"==========================================================")

        # Create a new model
        if torch.backends.mps.is_available():
            lr_schedule = linear_schedule(5e-4, 2.5e-6)
            clip_range_schedule = linear_schedule(0.150, 0.025)
            # instantiate a PPO agent using MPS (Apple Silicon, Metal Performance Shaders)
            model = MaskablePPO(
                "CnnPolicy",
                env,
                device="mps",
                verbose=1,
                n_steps=2048,
                batch_size=512*8,
                n_epochs=4,
                gamma=0.94,
                learning_rate=lr_schedule,
                clip_range=clip_range_schedule,
                tensorboard_log=LOG_DIR,
            )
        else:
            lr_schedule = linear_schedule(2.5e-4, 2.5e-6)
            clip_range_schedule = linear_schedule(0.150, 0.025)
            # instantiate a PPO agent using CUDA (NVIDIA GPU)
            model = MaskablePPO(
                "CnnPolicy",
                env,
                device="cuda",
                verbose=1,
                n_steps=1024,
                batch_size=512*4,
                n_epochs=4,
                gamma=0.94,
                learning_rate=lr_schedule,
                clip_range=clip_range_schedule,
                tensorboard_log=LOG_DIR,
            )

    checkpoint_interval = 15625 # checkpoint_interval * num_env = total_steps_per_checkpoint
    checkpoint_callback = CheckpointCallback(save_freq=checkpoint_interval, save_path=save_dir, name_prefix="ppo_tetris")

    original_stdout = sys.stdout
    log_file_path = os.path.join(save_dir, "training_log.txt")
    with open(log_file_path, 'w') as log_file:
        sys.stdout = log_file
        model.learn(total_timesteps=100_000_000, callback=checkpoint_callback)
        env.close()
    sys.stdout = original_stdout
    model.save(os.path.join(save_dir, "ppo_tetris_final.zip"))

if __name__ == "__main__":
    main()
