import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from stable_baselines3 import PPO
from GRF_Env import create_environment  # Shooter-only environment

def evaluate_shooter(render=True, trials=10):
    env = create_environment(render=render)
    model_shooter = PPO.load("Shooter/TrainingLog/Shooter_Training1", device='cpu')

    shooter_total = 0
    goals = {"left": 0, "center": 0, "right": 0}
    saves = {"left": 0, "center": 0, "right": 0}
    missed = 0

    ball_end_positions = []
    shot_directions = []
    

    for trial in range(trials):
        obs = env.reset()
        done = False
        episode_reward = 0
        shot_started = False
        shot_vector = None
        ball_logged = False

        while not done:
            if isinstance(obs, (list, tuple)):
                shooter_obs = obs[0]
            else:
                shooter_obs = obs

            shooter_obs = np.array(shooter_obs, dtype=np.float32).flatten()
            if shooter_obs.shape[0] < 94:
                raise ValueError(f"shooter_obs too short: {shooter_obs.shape[0]} values")

            ball_pos = shooter_obs[88:91]
            ball_vel = shooter_obs[91:94]

            # Detect shot start
            if not shot_started and np.linalg.norm(ball_vel[:2]) > 0.01:
                shot_started = True
                shot_vector = ball_vel[:2]

            # Detect ball bounce back (keeper save)
            if shot_started and prev_ball_pos is not None and not ball_logged:
                # If ball X decreases = bounced back
                if ball_pos[0] < prev_ball_pos[0]:
                    ball_end_positions.append(prev_ball_pos[:2])  # Log last forward position
                    if shot_vector is not None:
                        shot_directions.append(shot_vector)
                    ball_logged = True

            # Detect goal (ball crossed line)
            if not ball_logged and ball_pos[0] >= 1.00:
                ball_end_positions.append(ball_pos[:2])
                if shot_started and shot_vector is not None:
                    shot_directions.append(shot_vector)
                ball_logged = True

            prev_ball_pos = ball_pos.copy()  # Update previous ball position

            action_shooter, _ = model_shooter.predict(shooter_obs, deterministic=True)
            actions = [int(action_shooter)]

            step_result = env.step(actions)
            if len(step_result) == 4:
                obs, reward, done_flags, _ = step_result
                done = any(done_flags) if isinstance(done_flags, (list, np.ndarray)) else done_flags
            elif len(step_result) == 5:
                obs, reward, terminated, truncated, _ = step_result
                done = terminated or truncated
            else:
                raise RuntimeError("Unexpected number of outputs from env.step()")

            if isinstance(reward, (list, tuple)):
                episode_reward += reward[0]
            else:
                episode_reward += reward

            if render:
                env.render()

        shooter_total += episode_reward

        # Use reward as source of truth for goal
        goal_scored = episode_reward >= 1

        # Classify direction only if ball crossed line
        if ball_logged:
            y = ball_end_positions[-1][1]
            if goal_scored:
                if -0.04 <= y <= -0.02:
                    goals["left"] += 1
                elif -0.02 <= y <= 0.02:
                    goals["center"] += 1
                elif 0.02 <= y <= 0.04:
                    goals["right"] += 1
            else:
                if -0.04 <= y <= -0.02:
                    saves["left"] += 1
                elif -0.02 < y < 0.02:
                    saves["center"] += 1
                elif 0.02 <= y <= 0.04:
                    saves["right"] += 1
                else:
                    missed += 1
        else:
            missed += 1  # No valid final ball pos

        print(f"===> Trial {trial + 1} ended. Shooter Reward: {episode_reward:.2f}")

    print("\n✅ Evaluation Complete.")
    print(f"Average Shooter Reward: {shooter_total / trials:.2f}")
    print(f"Goals: {goals}")
    print(f"Saves: {saves}")
    print(f"Missed Shots: {missed}")

    # ==== PLOTTING ====
    os.makedirs("Shooter/Eval/Graphs", exist_ok=True)

    # Goal Directions
    plt.figure()
    categories = ["Left", "Center", "Right"]
    x = np.arange(len(categories))
    plt.bar(x - 0.2, [goals[c.lower()] for c in categories], width=0.4, label="Goals")
    plt.bar(x + 0.2, [saves[c.lower()] for c in categories], width=0.4, label="Saves")
    plt.xticks(x, categories)
    plt.title("Goals and Saves (PPO)")
    plt.ylabel("Count")
    plt.legend()
    plt.tight_layout()
    plt.savefig("Shooter/Eval/Graphs/goals_saves_direction_centre.png")
    plt.close()

    # Heatmap - Ball Position
    ball_array = np.array(ball_end_positions)
    if len(ball_array) > 0:
        plt.figure()
        sns.kdeplot(x=ball_array[:, 0], y=ball_array[:, 1], cmap="Reds", fill=True, bw_adjust=0.3, warn_singular=False)
        plt.title("Heatmap: Final Ball Position (PPO)")
        plt.xlabel("Ball X")
        plt.ylabel("Ball Y")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig("Shooter/Eval/Graphs/ball_position_heatmap_centre.png")
        plt.close()

    # Heatmap - Shot Direction
    shot_array = np.array(shot_directions)
    if len(shot_array) > 0:
        plt.figure()
        sns.kdeplot(x=shot_array[:, 0], y=shot_array[:, 1], cmap="Blues", fill=True, bw_adjust=0.4, warn_singular=False)
        plt.title("Heatmap: Shot Direction (Ball Velocity)")
        plt.xlabel("Vel X")
        plt.ylabel("Vel Y")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig("Shooter/Eval/Graphs/shot_direction_heatmap_centre.png")
        plt.close()

    env.close()

if __name__ == "__main__":
    evaluate_shooter(render=False, trials=1000)