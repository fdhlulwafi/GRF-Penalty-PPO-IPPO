import numpy as np
import time
from GRF_Env import create_environment

def test_keeper_wait_behavior_rendered(num_trials=10):
    print("===== Starting Keeper Behavior Test with Rendering =====")

    for trial in range(num_trials):
        print(f"\n--- Trial {trial + 1} ---")
        env = create_environment(render=True)
        obs = env.reset()
        done = False
        step_count = 0

        while not done and step_count < 200:
            actions = [np.random.randint(0, 19), np.random.randint(0, 19)]
            obs, reward, done, info = env.step(actions)

            env.render()  # Render the frame

            # Access ball_kicked from wrapper
            print(f"Step {step_count}: Ball kicked? {env.ball_kicked}, Reward: {reward}, Done: {done}")

            time.sleep(0.05)  # Slow down for better visual

            step_count += 1

        env.close()
        print(f"Trial {trial + 1} finished after {step_count} steps.")

    print("===== All Trials Completed =====")

if __name__ == "__main__":
    test_keeper_wait_behavior_rendered()
