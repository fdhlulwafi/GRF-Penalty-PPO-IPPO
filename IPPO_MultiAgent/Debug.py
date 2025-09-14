import traceback
import numpy as np
from Env import create_multiagent_env  # your multi-agent env creator
from SingleAgentWrapper import SingleAgentEnvWrapper

def test_agent_actions(agent_id, base_env):
    print(f"\nTesting actions for agent: {'Shooter' if agent_id == 0 else 'Keeper'}")
    wrapped_env = SingleAgentEnvWrapper(base_env, agent_id)

    # Patch step method to print received actions and their types
    original_step = wrapped_env.step

    def debug_step(actions):
        print(f"Step called with actions: {actions} (type: {type(actions)})")
        if isinstance(actions, (list, np.ndarray)):
            try:
                print(f"First element of actions: {actions[0]} (type: {type(actions[0])})")
            except Exception as e:
                print(f"Error accessing first element: {e}")
        try:
            result = original_step(actions)
            print(f"Step result: obs shape: {np.array(result[0]).shape}, reward: {result[1]}, done: {result[2]}")
            return result
        except Exception as e:
            print("Exception during step:")
            traceback.print_exc()
            raise e

    wrapped_env.step = debug_step

    obs = wrapped_env.reset()
    print(f"Initial observation shape: {np.array(obs).shape}")

    # Valid flat action (int) to test
    valid_action = 63  # example valid flat action for 19x19
    invalid_action = 67  # example invalid flat action (> 19*19-1)

    # Test valid action
    try:
        print(f"Testing valid action {valid_action}:")
        wrapped_env.step([valid_action])  # Note: passing list to simulate VecEnv style
    except Exception as e:
        print(f"Exception with valid action {valid_action}: {e}")

    # Test invalid action
    try:
        print(f"Testing invalid action {invalid_action}:")
        wrapped_env.step([invalid_action])
    except Exception as e:
        print(f"Expected exception with invalid action {invalid_action}: {e}")

if __name__ == "__main__":
    print("Creating multi-agent environment...")
    base_env = create_multiagent_env()
    print("Multi-agent environment created.")

    test_agent_actions(0, base_env)  # Shooter agent
    test_agent_actions(1, base_env)  # Keeper agent
