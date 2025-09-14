import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from GRF_Env import create_environment  # Shooter-only environment

def evaluate_kicking_force(render=False, trials=50):
    env = create_environment(render=render)
    model_shooter = PPO.load("Shooter/TrainingLog/Shooter_Training1", device='cpu')

    estimated_forces = []
    shot_directions = []
    goal_flags = []

    for trial in range(trials):
        obs = env.reset()
        done = False
        shot_started = False
        force_logged = False
        reward_total = 0

        prev_ball_vel = np.array([0.0, 0.0])
        initial_ball_vel = None

        while not done:
            if isinstance(obs, (list, tuple)):
                shooter_obs = obs[0]
            else:
                shooter_obs = obs

            shooter_obs = np.array(shooter_obs, dtype=np.float32).flatten()
            ball_vel = shooter_obs[91:93]  # Only x and y velocity (ignore z)

            # Detect shot start via velocity jump
            speed_now = np.linalg.norm(ball_vel)
            speed_prev = np.linalg.norm(prev_ball_vel)
            accel = speed_now - speed_prev

            if not shot_started and speed_now > 0.01 and accel > 0.01:
                shot_started = True
                initial_ball_vel = ball_vel.copy()

            if shot_started and not force_logged:
                # Log once when ball starts accelerating
                estimated_force = np.linalg.norm(initial_ball_vel)
                estimated_forces.append(estimated_force)
                shot_directions.append(initial_ball_vel)
                force_logged = True

            prev_ball_vel = ball_vel.copy()

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
                raise RuntimeError("Unexpected env.step() output")

            reward_val = reward[0] if isinstance(reward, (list, tuple)) else reward
            reward_total += reward_val

            if render:
                env.render()

        goal_flags.append(reward_total >= 1)
        print(f"===> Trial {trial+1}: {'GOAL' if reward_total >= 1 else 'MISS'} | Estimated Force: {estimated_forces[-1]:.4f}" if force_logged else f"===> Trial {trial+1}: NO SHOT DETECTED")

    env.close()

    # ===== PLOTTING =====

    os.makedirs("Shooter/Eval/Graphs", exist_ok=True)

    # Force Distribution Histogram
    if estimated_forces:
        plt.figure()
        plt.hist(estimated_forces, bins=15, color='skyblue', edgecolor='black')
        plt.title("Estimated Kicking Force Distribution")
        plt.xlabel("Initial Ball Velocity (Force Proxy)")
        plt.ylabel("Count")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig("Shooter/Eval/Graphs/force_distribution_left.png")
        plt.close()

        # Scatter plot: Force vs. Goal/Miss
        goal_flags = np.array(goal_flags[:len(estimated_forces)])
        forces = np.array(estimated_forces)

        plt.figure()
        plt.scatter(np.arange(len(forces))[goal_flags], forces[goal_flags], c='green', label='Goal', alpha=0.7)
        plt.scatter(np.arange(len(forces))[~goal_flags], forces[~goal_flags], c='red', label='Miss', alpha=0.7)
        plt.title("Kicking Force Comparison: Goal vs Miss (Left)")
        plt.xlabel("Trial Index")
        plt.ylabel("Estimated Force")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig("Shooter/Eval/Graphs/force_vs_goal_miss_left.png")
        plt.close()

    print("\n✅ Force Evaluation Complete.")
    print(f"Mean Estimated Force: {np.mean(estimated_forces):.4f}")
    print(f"Goals: {np.sum(goal_flags)} / {len(goal_flags)}")

if __name__ == "__main__":
    evaluate_kicking_force(render=False, trials=1000)
