import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import numpy as np
from stable_baselines3 import PPO
from GRF_Env import create_environment  # Make sure this creates the shooter-only env

def evaluate_shooter(render=True, trials=10):
    env = create_environment(render=render)
    model_shooter = PPO.load("Shooter/TrainingLog/Shooter_Training1", device='cpu')

    total_reward = 0

    for i in range(trials):
        obs = env.reset()
        done = False
        episode_reward = 0

        while not done:
            if isinstance(obs, (list, tuple)):
                shooter_obs = obs[0]
            else:
                shooter_obs = obs

            shooter_obs = np.array(shooter_obs, dtype=np.float32).flatten()
            action_shooter, _ = model_shooter.predict(shooter_obs, deterministic=True)
            actions = [int(action_shooter)]

            result = env.step(actions)
            if len(result) == 4:
                obs, reward, done_flags, _ = result
                done = any(done_flags) if isinstance(done_flags, (list, np.ndarray)) else done_flags
            elif len(result) == 5:
                obs, reward, terminated, truncated, _ = result
                done = terminated or truncated
            else:
                raise RuntimeError("Unexpected number of outputs from env.step()")

            if isinstance(reward, (list, tuple)):
                episode_reward += reward[0]
            else:
                episode_reward += reward

            if render:
                env.render()

        print(f"Trial {i + 1}: Shooter Reward = {episode_reward:.2f}")
        total_reward += episode_reward

    avg_reward = total_reward / trials
    print("\n✅ Evaluation Complete.")
    print(f"Total Reward: {total_reward:.2f}")
    print(f"Average Reward per Trial: {avg_reward:.2f}")

    env.close()

if __name__ == "__main__":
    evaluate_shooter(render=True, trials=5)
