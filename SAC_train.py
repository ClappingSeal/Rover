import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random
import math
import matplotlib.pyplot as plt
from stable_baselines3 import SAC
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import EvalCallback
from DRL_env import DRLEnv


def plot_learning_curve(log_dir, file_name='learning_curve_sac.png', averaging_window=100):
    results = np.load(log_dir + '/evaluations.npz')
    timesteps = results['timesteps']
    results_mean = results['results'].mean(axis=1)

    print(f"Timesteps shape: {timesteps.shape}")
    print(f"Results mean shape: {results_mean.shape}")

    if len(timesteps) < averaging_window:
        print(
            f"Averaging window ({averaging_window}) is larger than the number of timesteps ({len(timesteps)}). Reducing averaging window size.")
        averaging_window = len(timesteps)

    averaged_rewards = np.convolve(results_mean, np.ones(averaging_window) / averaging_window, mode='valid')
    adjusted_timesteps = timesteps[averaging_window - 1:]

    plt.plot(adjusted_timesteps, averaged_rewards)
    plt.xlabel('Timesteps')
    plt.ylabel('Mean Reward')
    plt.title('Learning Curve')
    plt.savefig(file_name)
    plt.show()


if __name__ == '__main__':
    log_dir = "./logs_sac/"

    env = DRLEnv()
    check_env(env)

    eval_env = DRLEnv()
    eval_callback = EvalCallback(eval_env, best_model_save_path=log_dir,
                                 log_path=log_dir, eval_freq=1000,
                                 deterministic=True, render=False)

    # n_steps removed and other hyperparameters like buffer_size are added
    model = SAC("MlpPolicy", env, verbose=1, buffer_size=100000, batch_size=256, learning_rate=3e-4)
    model.learn(total_timesteps=10000000, callback=eval_callback)
    model.save("sac_3")
    plot_learning_curve(log_dir, file_name='learning_curve_sac.png', averaging_window=100)
