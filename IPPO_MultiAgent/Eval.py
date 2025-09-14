import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from stable_baselines3 import PPO
from Env import create_multiagent_env  # Your multi-agent environment

def evaluate_agents(render=True, trials=10):
    env = create_multiagent_env(render=render)

    model_shooter = PPO.load("IPPO_MultiAgent/MultiAgentTraining/TrainingLog/shooter_model_8", device='cpu')
    model_keeper = PPO.load("IPPO_MultiAgent/MultiAgentTraining/TrainingLog/keeper_model_8", device='cpu')

    shooter_total = 0
    keeper_total = 0

    # Stats
    goals = {"left": 0, "center": 0, "right": 0}
    saves = {"left": 0, "center": 0, "right": 0}
    missed = 0
    fouls = 0

    ball_end_positions = []
    shot_directions = []
    keeper_movements = []

    for trial in range(trials):
        obs = env.reset()
        done = False
        episode_reward = [0, 0]
        shot_started = False
        keeper_moved_early = False
        ball_logged = False  # <--- NEW: Flag to prevent multiple logging

        initial_keeper_pos = None
        keeper_end_pos = None
        initial_ball_pos = None
        initial_ball_vel = None

        while not done:
            shooter_obs = obs[0]
            keeper_obs = obs[1]

            ball_pos = np.array([shooter_obs[88], shooter_obs[89], shooter_obs[90]])
            ball_vel = np.array([shooter_obs[91], shooter_obs[92], shooter_obs[93]])
            keeper_pos = np.array([keeper_obs[0], keeper_obs[1]])

            if initial_keeper_pos is None:
                initial_keeper_pos = keeper_pos.copy()
            if initial_ball_vel is None:
                initial_ball_vel = ball_vel[:2].copy()
            if initial_ball_pos is None:
                initial_ball_pos = ball_pos[:2].copy()

            if not shot_started and np.linalg.norm(ball_vel[:2]) > 0.01:
                shot_started = True
                shot_vector = ball_vel[:2]

            if not shot_started and np.linalg.norm(keeper_pos - initial_keeper_pos) > 0.01:
                keeper_moved_early = True

            # 🟢 NEW: log final ball pos once it crosses x=1.00
            if not ball_logged and ball_pos[0] >= 1.00:
                ball_end_positions.append(ball_pos[:2])
                if shot_started:
                    shot_directions.append(shot_vector)
                if initial_keeper_pos is not None:
                    keeper_end_pos = keeper_pos.copy()
                    keeper_movements.append(keeper_end_pos - initial_keeper_pos)
                ball_logged = True

            action_shooter, _ = model_shooter.predict(shooter_obs, deterministic=True)
            action_keeper, _ = model_keeper.predict(keeper_obs, deterministic=True)
            actions = [int(action_shooter), int(action_keeper)]

            step_result = env.step(actions)
            if len(step_result) == 4:
                obs, reward, done_flags, info = step_result
                done = any(done_flags) if isinstance(done_flags, (list, np.ndarray)) else done_flags
            elif len(step_result) == 5:
                obs, reward, terminated, truncated, info = step_result
                done = terminated or truncated
            else:
                raise RuntimeError("Unexpected number of outputs from env.step()")

            episode_reward[0] += reward[0]
            episode_reward[1] += reward[1]

            if render:
                env.render()

        shooter_total += episode_reward[0]
        keeper_total += episode_reward[1]
        keeper_end_pos = keeper_pos.copy()

        # Outcome classification
        x, y = ball_pos[0], ball_pos[1]
        goal_scored = episode_reward[0] >= 1

        if keeper_moved_early:
            fouls += 1

        if goal_scored:
            if -0.04 <= y <= -0.02:
                goals["left"] += 1
            elif -0.02 <= y <= 0.02:
                goals["center"] += 1
            elif 0.02 <= y <= 0.04:
                goals["right"] += 1
        else:
            # Ball didn't result in goal — could be save or miss
            if -0.04 <= y <= 0.02:
                saves["left"] += 1
            elif -0.02 < y < 0.02:
                saves["center"] += 1
            elif 0.02 <= y <= 0.04:
                saves["right"] += 1
            else:
                missed += 1

        # Track when the ball reaches or crosses the goal line (x ≈ 1.00)
        if not ball_logged and ball_pos[0] >= 1.00:
            ball_end_positions.append(ball_pos[:2])
            if shot_started:
                shot_directions.append(shot_vector)
            if initial_keeper_pos is not None and keeper_end_pos is not None:
                keeper_movements.append(keeper_end_pos - initial_keeper_pos)
            ball_logged = True  # Prevent logging multiple times


        print(f"===> Trial {trial + 1} ended. Shooter Reward: {episode_reward[0]:.2f}, Keeper Reward: {episode_reward[1]:.2f}")

    print("\n✅ Evaluation Complete.")
    print(f"Average Shooter Reward: {shooter_total / trials:.2f}")
    print(f"Average Keeper Reward: {keeper_total / trials:.2f}")
    print(f"Goals: {goals}")
    print(f"Saves: {saves}")
    print(f"Missed Shots: {missed}")
    print(f"Fouls (Early Keeper Moves): {fouls}")

    # ==== PLOTTING ====
    os.makedirs("IPPO_MultiAgent/Eval/Graphs", exist_ok=True)

    # Goal/Save per direction
    plt.figure()
    categories = ["Left", "Center", "Right"]
    x = np.arange(len(categories))
    plt.bar(x - 0.2, [goals[c.lower()] for c in categories], width=0.4, label="Goals")
    plt.bar(x + 0.2, [saves[c.lower()] for c in categories], width=0.4, label="Saves")
    plt.xticks(x, categories)
    plt.title("Goals and Saves (IPPO)")
    plt.ylabel("No. of Goal")
    plt.legend()
    plt.tight_layout()
    plt.savefig("IPPO_MultiAgent/Eval/Graphs/goals_saves_direction3.png")
    plt.close()

    # Heatmap - Ball Position
    ball_array = np.array(ball_end_positions)
    if len(ball_array) > 0:
        plt.figure()
        sns.kdeplot(x=ball_array[:, 0], y=ball_array[:, 1], cmap="Reds", fill=True, bw_adjust=0.3, warn_singular=False)
        plt.title("Heatmap: Final Ball Position (IPPO)")
        plt.xlabel("Ball X")
        plt.ylabel("Ball Y")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig("IPPO_MultiAgent/Eval/Graphs/ball_position_heatmap3.png")
        plt.close()

    # Heatmap - Shot Direction
    shot_array = np.array(shot_directions)
    if len(shot_array) > 0:
        plt.figure()
        sns.kdeplot(x=shot_array[:, 0], y=shot_array[:, 1], cmap="Blues", fill=True, bw_adjust=0.4, warn_singular=False)
        plt.title("Heatmap: Velocity of the Ball (IPPO)")
        plt.xlabel("Vel X")
        plt.ylabel("Vel Y")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig("IPPO_MultiAgent/Eval/Graphs/shot_direction_heatmap3.png")
        plt.close()

    # Heatmap - Keeper Movement
    keeper_array = np.array(keeper_movements)
    if len(keeper_array) > 0:
        plt.figure()
        sns.kdeplot(x=keeper_array[:, 0], y=keeper_array[:, 1], cmap="Greens", fill=True, bw_adjust=0.4, warn_singular=False)
        plt.title("Heatmap: Keeper Movement")
        plt.xlabel("ΔX")
        plt.ylabel("ΔY")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig("IPPO_MultiAgent/Eval/Graphs/keeper_movement_heatmap3.png")
        plt.close()

    env.close()


if __name__ == "__main__":
    evaluate_agents(render=False, trials=1000)
