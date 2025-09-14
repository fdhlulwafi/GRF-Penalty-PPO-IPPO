import os
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from Env import create_multiagent_env
from SingleAgentWrapper import SingleAgentEnvWrapper

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

device = "cpu"  # or "cuda"
policy_kwargs = dict(net_arch=[256, 256])

total_timesteps = 500000
cycle_steps = 16384

def evaluate(model, env, episodes=10):
    total_reward = 0
    for _ in range(episodes):
        obs = env.reset()
        done = [False]
        while not done[0]:
            action, _ = model.predict(obs, deterministic=False)
            obs, reward, done, _ = env.step(action)
            total_reward += reward[0]
    mean_reward = total_reward / episodes
    return mean_reward

# Initialize environments
print("🎯 Initializing models and environments...")

shooter_wrapper = SingleAgentEnvWrapper(create_multiagent_env(), agent_id=0)
keeper_wrapper = SingleAgentEnvWrapper(create_multiagent_env(), agent_id=1)

vec_shooter_env = DummyVecEnv([lambda: shooter_wrapper])
vec_keeper_env = DummyVecEnv([lambda: keeper_wrapper])

model_shooter = PPO(
    "MlpPolicy",
    vec_shooter_env,
    policy_kwargs=policy_kwargs,
    verbose=1,
    learning_rate=3e-4,
    n_steps=8192,
    batch_size=256,
    n_epochs=20,
    gamma=0.995,
    gae_lambda=0.97,
    clip_range=0.2,
    ent_coef=0.01,
    vf_coef=0.5,
    max_grad_norm=0.5,
    tensorboard_log="IPPO_MultiAgent/MultiAgentTraining/TB/Shooter_TB8/",
    device=device,
)

model_keeper = PPO(
    "MlpPolicy",
    vec_keeper_env,
    policy_kwargs=policy_kwargs,
    verbose=1,
    learning_rate=5e-4,
    n_steps=8192,
    batch_size=256,
    n_epochs=10,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.3,
    ent_coef=0.05,
    vf_coef=0.4,
    max_grad_norm=0.5,
    tensorboard_log="IPPO_MultiAgent/MultiAgentTraining/TB/Keeper_TB8/",
    device=device,
)

# Training loop
print("\n🔁 Starting cyclical training...")

step_shooter = 0
step_keeper = 0
shooter_rewards = []
keeper_rewards = []
shooter_steps = []
keeper_steps = []

while step_keeper < total_timesteps:
    # 🔁 Update shooter env to use current keeper
    shooter_wrapper.set_other_agent_model(model_keeper)

    # 🎯 Train Shooter
    print(f"\n🎯 Training SHOOTER from step {step_shooter} to {step_shooter + cycle_steps}")
    model_shooter.learn(total_timesteps=cycle_steps, reset_num_timesteps=False)
    model_shooter.save("IPPO_MultiAgent/MultiAgentTraining/TrainingLog/shooter_model_8")

    shooter_reward = evaluate(model_shooter, vec_shooter_env)
    print(f"✅ Shooter Reward: {shooter_reward:.2f}")
    shooter_rewards.append(shooter_reward)
    shooter_steps.append(step_shooter + cycle_steps)
    step_shooter += cycle_steps

    if step_shooter >= total_timesteps:
        break

    # 🔁 Update keeper env to use current shooter
    keeper_wrapper.set_other_agent_model(model_shooter)

    # 🧤 Train Keeper
    print(f"\n🧤 Training KEEPER from step {step_keeper} to {step_keeper + cycle_steps}")
    model_keeper.learn(total_timesteps=cycle_steps, reset_num_timesteps=False)
    model_keeper.save("IPPO_MultiAgent/MultiAgentTraining/TrainingLog/keeper_model_8")

    keeper_reward = evaluate(model_keeper, vec_keeper_env)
    print(f"✅ Keeper Reward: {keeper_reward:.2f}")
    keeper_rewards.append(keeper_reward)
    keeper_steps.append(step_keeper + cycle_steps)
    step_keeper += cycle_steps

# Final log
print("\n✅ Training complete. Final models saved.")
print("\n📊 Visualize logs with:")
print("   tensorboard --logdir IPPO_MultiAgent/MultiAgentTraining/TB")

# === Save Graphs ===
os.makedirs("IPPO_MultiAgent/MultiAgentTraining/TrainingGraph", exist_ok=True)

# Shooter graph
plt.figure(figsize=(8, 5))
plt.plot(shooter_steps, shooter_rewards, marker='o', color='blue')
plt.xlabel("Training Steps")
plt.ylabel("Shooter Mean Reward")
plt.title("Shooter Training Reward Progression")
plt.grid(True)
plt.tight_layout()
shooter_plot_path = "IPPO_MultiAgent/MultiAgentTraining/TrainingGraph/shooter_reward_progression_8.png"
plt.savefig(shooter_plot_path)
print(f"📈 Shooter reward graph saved to: {shooter_plot_path}")
plt.close()

# Keeper graph
plt.figure(figsize=(8, 5))
plt.plot(keeper_steps, keeper_rewards, marker='x', color='green')
plt.xlabel("Training Steps")
plt.ylabel("Keeper Mean Reward")
plt.title("Keeper Training Reward Progression")
plt.grid(True)
plt.tight_layout()
keeper_plot_path = "IPPO_MultiAgent/MultiAgentTraining/TrainingGraph/keeper_reward_progression_8.png"
plt.savefig(keeper_plot_path)
print(f"📈 Keeper reward graph saved to: {keeper_plot_path}")
plt.close()