import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for plotting
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from GRF_Env import create_environment
import numpy as np

def test_keeper_with_render_and_plot():
    try:
        env = create_environment()

        model = PPO.load("TrainingLog/Shooter_Training3.zip", device='cpu')

        max_trials = 10
        trial_count = 0

        while trial_count < max_trials:
            obs = env.reset()
            done = False
            ball_kicked = False

            ball_positions_x = []
            ball_positions_y = []
            ball_velocities = []
            trial_rewards = []

            print(f"\n=== Trial {trial_count + 1} ===")

            step_num = 0
            while not done:
                env.render()
                env.render(mode="rgb_array")  # For completeness

                # Extract ball position and velocity from obs
                ball_pos = obs[88:91]       # x, y, z
                ball_vel = obs[91:94]       # x, y, z

                ball_positions_x.append(ball_pos[0])
                ball_positions_y.append(ball_pos[1])
                ball_velocities.append(np.linalg.norm(ball_vel))

                # Detect if ball is kicked
                if not ball_kicked and (ball_pos[0] > 0.811 or ball_pos[0] < 0.799):
                    ball_kicked = True
                    print("Ball has been kicked.")
                    
                action, _ = model.predict(obs)
                action = np.asarray(action)

                obs, reward, done, info = env.step(action)

                print("Game mode:", info.get("game_mode", "Not available"))

                print(f"Step {step_num}: Ball Pos=({ball_pos[0]:.3f}, {ball_pos[1]:.3f}, {ball_pos[2]:.3f}), "
                      f"Velocity={np.linalg.norm(ball_vel):.3f}, Action={action}")
                step_num += 1

            # Plot results
            plt.figure(figsize=(8, 6))
            plt.suptitle(f"Trial {trial_count + 1} Ball Position & Velocity")

            plt.subplot(2, 1, 1)
            plt.plot(ball_positions_x, label="X Position")
            plt.plot(ball_positions_y, label="Y Position")
            plt.ylabel("Position")
            plt.legend()
            plt.grid(True)

            plt.subplot(2, 1, 2)
            plt.plot(ball_velocities, label="Velocity Magnitude", color='red')
            plt.xlabel("Step")
            plt.ylabel("Velocity")
            plt.legend()
            plt.grid(True)

            os.makedirs("MotionLog", exist_ok=True)
            plot_path = f"MotionLog/BallMotion_Trial{trial_count + 1}.png"
            plt.savefig(plot_path)
            plt.close()
            print(f"Saved plot to {plot_path}")

            trial_count += 1

    except Exception as e:
        print(f"An error occurred: {str(e)}")

    finally:
        env.close()

if __name__ == "__main__":
    test_keeper_with_render_and_plot()
