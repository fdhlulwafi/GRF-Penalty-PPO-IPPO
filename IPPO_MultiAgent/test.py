import os
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from Env import create_multiagent_env  # Your multi-agent env creator
from SingleAgentWrapper import SingleAgentEnvWrapper  # Your single-agent wrapper

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

device = "cpu"  # or "cuda"

# Create base multi-agent environment once
base_env = create_multiagent_env()

# Factory functions for single-agent wrapped envs
def make_shooter_env():
    # Wrap base_env with agent_id=0 shooter
    return SingleAgentEnvWrapper(base_env, agent_id=0)

def make_keeper_env():
    # Wrap base_env with agent_id=1 keeper
    return SingleAgentEnvWrapper(base_env, agent_id=1)

# Wrap single-agent envs in DummyVecEnv (vectorized envs with a single copy)
vec_shooter_env = DummyVecEnv([make_shooter_env])
vec_keeper_env = DummyVecEnv([make_keeper_env])

policy_kwargs = dict(net_arch=[256, 256])

model_shooter = PPO(
    "MlpPolicy",
    vec_shooter_env,
    policy_kwargs=policy_kwargs,
    verbose=1,
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=256,
    n_epochs=20,
    gamma=0.995,
    gae_lambda=0.97,
    clip_range=0.2,
    ent_coef=0.01,
    vf_coef=0.5,
    max_grad_norm=0.5,
    tensorboard_log="MultiAgentTraining2/shooter_tb/",
    device=device,
)

model_keeper = PPO(
    "MlpPolicy",
    vec_keeper_env,
    policy_kwargs=policy_kwargs,
    verbose=1,
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=256,
    n_epochs=20,
    gamma=0.995,
    gae_lambda=0.97,
    clip_range=0.2,
    ent_coef=0.01,
    vf_coef=0.5,
    max_grad_norm=0.5,
    tensorboard_log="MultiAgentTraining2/keeper_tb/",
    device=device,
)

total_timesteps = 100_000
rollout_steps = 2048

# Initialize last other agent action for wrappers
vec_shooter_env.envs[0].set_other_agent_action(0)
vec_keeper_env.envs[0].set_other_agent_action(0)

# Reset both wrapped envs to get initial observations
obs_shooter = vec_shooter_env.reset()
obs_keeper = vec_keeper_env.reset()

total_shooter_reward = 0
total_keeper_reward = 0

for step in range(0, total_timesteps, rollout_steps):
    for _ in range(rollout_steps):
        # Predict actions for each agent (arrays of shape (1,))
        action_shooter, _ = model_shooter.predict(obs_shooter, deterministic=False)
        action_keeper, _ = model_keeper.predict(obs_keeper, deterministic=False)

        # Convert to int scalars
        action_shooter = int(action_shooter.flatten()[0])
        action_keeper = int(action_keeper.flatten()[0])

        # Sync other agent actions inside wrappers
        vec_shooter_env.envs[0].set_other_agent_action(action_keeper)
        vec_keeper_env.envs[0].set_other_agent_action(action_shooter)

        # Step each wrapped env separately with the agent's action
        obs_shooter, reward_shooter, done_shooter, info_shooter = vec_shooter_env.step([action_shooter])
        obs_keeper, reward_keeper, done_keeper, info_keeper = vec_keeper_env.step([action_keeper])

        total_shooter_reward += reward_shooter[0]
        total_keeper_reward += reward_keeper[0]

        # If either agent done, reset both environments and base_env together
        if done_shooter[0] or done_keeper[0]:
            # Reset base_env once, then reset wrappers
            base_env.reset()
            obs_shooter = vec_shooter_env.reset()
            obs_keeper = vec_keeper_env.reset()
            vec_shooter_env.envs[0].set_other_agent_action(0)
            vec_keeper_env.envs[0].set_other_agent_action(0)

    print(f"Step {step} - Episode Reward: Shooter={total_shooter_reward:.2f}, Keeper={total_keeper_reward:.2f}")

    # Train both agents on their collected rollouts
    model_shooter.learn(total_timesteps=rollout_steps, reset_num_timesteps=False)
    model_keeper.learn(total_timesteps=rollout_steps, reset_num_timesteps=False)

# Save models
model_shooter.save("MultiAgentTraining2/shooter_model")
model_keeper.save("MultiAgentTraining2/keeper_model")

base_env.close()

print("✅ Training complete. Use TensorBoard to visualize logs:")
print("   tensorboard --logdir MultiAgentTraining2")