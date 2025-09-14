import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from GRF_Env import create_environment  # Shooter-only environment

def evaluate_ball_trajectory(render=False, trials=5):
    env = create_environment(render=render)
    model_shooter = PPO.load("Shooter/TrainingLog/Shooter_Training1", device='cpu')

    os.makedirs("Shooter/Eval/Graphs", exist_ok=True)

    all_trajectories = []  # List of trajectories: each is a list of (x, y)
    results = []

    for trial in range(trials):
        obs = env.reset()
        done = False
        trajectory = []
        shot_started = False
        prev_ball_pos = None
        reward_total = 0
        reward_reached = False

        while not done:
            if isinstance(obs, (list, tuple)):
                shooter_obs = obs[0]
            else:
                shooter_obs = obs

            shooter_obs = np.array(shooter_obs, dtype=np.float32).flatten()
            ball_pos = shooter_obs[88:91]
            ball_vel = shooter_obs[91:94]

            # Track ball trajectory
            if not reward_reached:
                trajectory.append(ball_pos[:2].copy())

            if not shot_started and np.linalg.norm(ball_vel[:2]) > 0.01:
                shot_started = True

            action, _ = model_shooter.predict(shooter_obs, deterministic=True)
            actions = [int(action)]

            step_result = env.step(actions)
            if len(step_result) == 4:
                obs, reward, done_flags, _ = step_result
                done = any(done_flags) if isinstance(done_flags, (list, np.ndarray)) else done_flags
            elif len(step_result) == 5:
                obs, reward, terminated, truncated, _ = step_result
                done = terminated or truncated
            else:
                raise RuntimeError("Unexpected step result from env.step()")

            reward_val = reward[0] if isinstance(reward, (list, tuple)) else reward
            reward_total += reward_val

            # Stop tracking when goal scored
            if not reward_reached and (reward_val >= 1 or ball_pos[0] >= 1.0):
                reward_reached = True

            if render:
                env.render()

        # Classify result
        final_x, final_y = trajectory[-1]
        if reward_total >= 1 or final_x >= 1.0:
            results.append("goal")
        elif any(trajectory[i][0] > trajectory[i+1][0] for i in range(len(trajectory)-1)):
            results.append("save")
        else:
            results.append("miss")

        all_trajectories.append(trajectory)
        print(f"===> Trial {trial+1}: {results[-1].upper()} | Reward: {reward_total:.2f}")

    env.close()

    # Plot all trajectories with different colors
    plt.figure(figsize=(8, 5))
    colors = plt.cm.get_cmap("tab10", trials)

    for i, (trajectory, result) in enumerate(zip(all_trajectories, results)):
        trajectory = np.array(trajectory)
        plt.plot(trajectory[:, 0], trajectory[:, 1], color=colors(i), label=f"Trial {i+1} ({result})")

    plt.axvline(x=1.0, color='black', linestyle='--', label='Goal Line')
    plt.title("Ball Trajectories (Right)")
    plt.xlabel("Ball X Position")
    plt.ylabel("Ball Y Position")
    plt.grid(True)
    plt.xlim(0.8, 1.2)
    plt.ylim(-0.06, 0.06)

    plt.tight_layout()
    plt.savefig("Shooter/Eval/Graphs/ball_trajectory_lines_Right.png")
    plt.close()

if __name__ == "__main__":
    evaluate_ball_trajectory(render=False, trials=1000)
