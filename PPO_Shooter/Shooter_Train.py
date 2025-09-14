import os
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from GRF_Env import create_environment  # Ensure this returns a Gym-like env

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

device = "cpu"  # or "cuda"
policy_kwargs = dict(net_arch=[256, 256])

total_timesteps = 500000
cycle_steps = 16384

def evaluate(model, episodes=10):
    total_reward = 0
    env = create_environment()  # Create evaluation env here
    for _ in range(episodes):
        obs = env.reset()
        done = False
        episode_reward = 0
        while not done:
            action, _ = model.predict(obs, deterministic=False)
            obs, reward, done, _ = env.step(action)
            episode_reward += reward
        total_reward += episode_reward
    env.close()
    return total_reward / episodes

# Initialize environment for training
print("🎯 Initializing models and environments...")
env = create_environment()  # <- Fix: create environment before PPO

# Initialize PPO model
model_shooter = PPO(
    "MlpPolicy",
    env,  # <- Fix: pass env here
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
    tensorboard_log="Shooter/TB/Shooter_TB1/",
    device=device,
)

# Training loop
print("\n🔁 Starting cyclical training...")

step_shooter = 0
shooter_rewards = []
shooter_steps = []

while step_shooter < total_timesteps:
    print(f"\n🎯 Training SHOOTER from step {step_shooter} to {step_shooter + cycle_steps}")
    model_shooter.learn(total_timesteps=cycle_steps, reset_num_timesteps=False)
    model_shooter.save("Shooter/TrainingLog/Shooter_Training1")

    shooter_reward = evaluate(model_shooter)
    print(f"✅ Shooter Reward: {shooter_reward:.2f}")
    shooter_rewards.append(shooter_reward)
    shooter_steps.append(step_shooter + cycle_steps)
    step_shooter += cycle_steps

# Final log
print("\n✅ Training complete. Final models saved.")
print("\n📊 Visualize logs with:")
print("   tensorboard --logdir PPO_TensorBoard")

# === Save Graphs ===
os.makedirs("Shooter/TrainingGraph", exist_ok=True)

# Shooter graph
plt.figure(figsize=(8, 5))
plt.plot(shooter_steps, shooter_rewards, marker='o', color='blue')
plt.xlabel("Training Steps")
plt.ylabel("Shooter Mean Reward")
plt.title("Shooter Training Reward Progression")
plt.grid(True)
plt.tight_layout()
shooter_plot_path = "Shooter/TrainingGraph/shooter_reward_progression_8.png"
plt.savefig(shooter_plot_path)
print(f"📈 Shooter reward graph saved to: {shooter_plot_path}")
plt.close()
