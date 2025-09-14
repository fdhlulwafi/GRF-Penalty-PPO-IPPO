import gfootball.env as football_env
from PenaltyWrapper import KeeperWaitWrapper

def create_environment(render=False):
    env = football_env.create_environment(
        env_name="penalty_shooter",
        representation="simple115v2",
        number_of_left_players_agent_controls=1,
        # number_of_right_players_agent_controls=1,
        render=render,
        # any other direct arguments your version supports
    )
    # env = KeeperWaitWrapper(env)
    return env
