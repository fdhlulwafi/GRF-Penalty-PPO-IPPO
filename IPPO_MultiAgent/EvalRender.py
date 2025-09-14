import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from stable_baselines3 import PPO
from Env import create_multiagent_env  # Your multi-agent environment
import numpy as np

def evaluate_agents(render=True, trials=1):
    env = create_multiagent_env(render=render)

    model_shooter = PPO.load("IPPO_MultiAgent/MultiAgentTraining/TrainingLog/shooter_model_7", device='cpu')
    model_keeper = PPO.load("IPPO_MultiAgent/MultiAgentTraining/TrainingLog/keeper_model_7", device='cpu')

    shooter_total = 0
    keeper_total = 0

    for trial in range(trials):
        obs = env.reset()
        done = False
        episode_reward = [0, 0]  # For shooter and keeper

        while not done:
            shooter_obs = obs[0]
            keeper_obs = obs[1]

            # Extract positions
            ball_pos = np.array([shooter_obs[88], shooter_obs[89], shooter_obs[90]])
            keeper_pos = np.array([keeper_obs[0], keeper_obs[1]])

            ball_pos_str = f"({ball_pos[0]:.4f}, {ball_pos[1]:.4f}, {ball_pos[2]:.4f})"

            action_shooter, _ = model_shooter.predict(shooter_obs, deterministic=True)
            action_keeper, _ = model_keeper.predict(keeper_obs, deterministic=True)

            actions = [int(action_shooter), int(action_keeper)]

            step_result = env.step(actions)
            if len(step_result) == 4:
                obs, reward, done_flags, info = step_result
                if isinstance(done_flags, (list, np.ndarray)):
                    done = any(done_flags)
                else:
                    done = done_flags
            elif len(step_result) == 5:
                obs, reward, terminated, truncated, info = step_result
                done = terminated or truncated
            else:
                raise RuntimeError("Unexpected number of outputs from env.step()")

            episode_reward[0] += reward[0]
            episode_reward[1] += reward[1]

            print(f"Trial {trial + 1} | Ball Pos: {ball_pos_str} |  Rewards: Shooter={reward[0]:.2f}, Keeper={reward[1]:.2f} | Actions: {actions}")
            if render:
                env.render()

        shooter_total += episode_reward[0]
        keeper_total += episode_reward[1]
        print(f"===> Trial {trial + 1} ended. Total Episode Reward - Shooter: {episode_reward[0]:.2f}, Keeper: {episode_reward[1]:.2f}\n")

    avg_shooter = shooter_total / trials
    avg_keeper = keeper_total / trials

    print("✅ Evaluation Complete.")
    print(f"Average Reward over {trials} Trials - Shooter: {avg_shooter:.2f}, Keeper: {avg_keeper:.2f}")

    env.close()

if __name__ == "__main__":
    evaluate_agents()