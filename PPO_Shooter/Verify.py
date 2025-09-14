from GRF_Env import create_environment

env = create_environment(render=False)
obs = env.reset()
done = False

step_count = 0
while not done and step_count < 5:
    action_space = env.action_space
    try:
        # Try sampling for both agents (multi-agent mode)
        action = [action_space[0].sample(), action_space[1].sample()]
    except:
        # Fallback if single-agent
        action = action_space.sample()

    obs, reward, done, info = env.step(action)
    
    print(f"\nStep {step_count}")
    print(f"Type of reward: {type(reward)}")
    print(f"Reward: {reward}")
    print(f"Observation type: {type(obs)}")
    print(f"Done: {done}")
    
    step_count += 1
