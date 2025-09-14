import gfootball.env as football_env
from RewardWrapper import RewardWrapper  # Keep your reward shaping wrapper

def create_multiagent_env(render=False):
    env = football_env.create_environment(
        env_name="11_vs_11_competition",
        representation="simple115v2",
        number_of_left_players_agent_controls=1,
        number_of_right_players_agent_controls=1,
        render=render,
    )

    # Apply keeper reward wrapper (keeper_agent_id=1, shooter_agent_id=0)
    env = RewardWrapper(env, keeper_agent_id=1, shooter_agent_id=0)

    return env
