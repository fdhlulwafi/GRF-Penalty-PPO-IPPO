import gym
import numpy as np

class RewardWrapper(gym.Wrapper):
    def __init__(self, env, keeper_agent_id=1, shooter_agent_id=0):
        super().__init__(env)
        self.keeper_agent_id = keeper_agent_id
        self.shooter_agent_id = shooter_agent_id

        self.ball_kicked = False
        self.goal_reward_given = False

        self.previous_ball_pos = None
        self.previous_keeper_pos = None
        self.previous_ball_y = None
        self.previous_keeper_action = None

        self.steps_after_kick = 0
        self.y_reward_given = False

    def step(self, actions):
        obs, rewards, done, info = self.env.step(actions)

        # Extract relevant positions
        ball_pos = self._extract_ball_pos(obs[self.shooter_agent_id])
        keeper_pos = self._extract_keeper_pos(obs[self.keeper_agent_id])

        # Detect ball kick
        if self.previous_ball_pos is None:
            self.previous_ball_pos = ball_pos.copy()
        else:
            ball_movement = np.linalg.norm(ball_pos - self.previous_ball_pos)
            if not self.ball_kicked and ball_movement > 0.01:
                self.ball_kicked = True
            self.previous_ball_pos = ball_pos.copy()

        shooter_reward = rewards[self.shooter_agent_id]

        # Track previous keeper position
        if self.previous_keeper_pos is None:
            self.previous_keeper_pos = keeper_pos.copy()
        prev_keeper_pos = self.previous_keeper_pos
        self.previous_keeper_pos = keeper_pos.copy()

        # ---------------- REWARD SHAPING FOR KEEPER ----------------

        # 1️⃣ Goal scored → -3 penalty to keeper
        if shooter_reward == 1.0 and not self.goal_reward_given:
            rewards[self.keeper_agent_id] += -3.0
            self.goal_reward_given = True
            return obs, rewards, done, info

        # 2️⃣ Ball blocked → +3 reward to keeper
        if ball_pos[0] < self.previous_ball_pos[0] and not self.goal_reward_given:
            rewards[self.keeper_agent_id] += 3.0
            self.goal_reward_given = True
            return obs, rewards, done, info
        
        # Missed shot → ball x > 1.0 and shooter didn't score
        if ball_pos[0] > 1.0 and shooter_reward != 0.0 and not self.goal_reward_given:
            self.goal_reward_given = True
            return obs, rewards, done, info

        # 🔒 Block all shaping after goal was already rewarded
        if self.goal_reward_given:
            return obs, rewards, done, info

        # 3️⃣ Before kick: reward stillness, penalize movement
        if not self.ball_kicked:
            if actions[self.keeper_agent_id] == 0:  # stay still
                rewards[self.keeper_agent_id] += 0.01
            else:
                rewards[self.keeper_agent_id] += -0.1
            self.steps_after_kick = 0
        else:
            self.steps_after_kick += 1

            # 🕒 Time penalty to discourage delay
            rewards[self.keeper_agent_id] += -0.01

            # ✅ Quick first reaction
            if self.steps_after_kick == 1 and actions[self.keeper_agent_id] != 0:
                rewards[self.keeper_agent_id] += 0.05

            # ✅ Early commitment in first 3 steps
            if self.steps_after_kick <= 3 and actions[self.keeper_agent_id] in [3, 5, 7]:
                rewards[self.keeper_agent_id] += 0.05

            # ⚠️ Penalize wrong dive direction
            def get_expected_dive(ball_y):
                if ball_y > 0.01:
                    return 3  # right dive
                elif ball_y < -0.01:
                    return 7  # left dive
                else:
                    return 5  # forward

            expected_action = get_expected_dive(ball_pos[1])
            if actions[self.keeper_agent_id] != expected_action:
                rewards[self.keeper_agent_id] += -0.5  # softer penalty

            # ✅ Encourage movement
            if actions[self.keeper_agent_id] != 0:
                rewards[self.keeper_agent_id] += 0.01
            else:
                rewards[self.keeper_agent_id] += -0.05

            # 🎯 Y-alignment reward (once only)
            if not self.y_reward_given and abs(keeper_pos[1] - ball_pos[1]) < 0.01:
                rewards[self.keeper_agent_id] += 0.05
                self.y_reward_given = True

            # 🧤 Close block near goal
            if ball_pos[0] > 0.9 and np.linalg.norm(keeper_pos - ball_pos) < 0.05:
                rewards[self.keeper_agent_id] += 0.1

            # 🦶 Encourage approach
            distance_to_ball = np.linalg.norm(keeper_pos - ball_pos)
            prev_distance = np.linalg.norm(prev_keeper_pos - ball_pos)
            if prev_distance - distance_to_ball > 0.02 and actions[self.keeper_agent_id] in [9, 10, 11, 12, 16]:
                rewards[self.keeper_agent_id] += 0.1
            else:
                rewards[self.keeper_agent_id] += -0.1

            # 🤸 Reward full-body dive for wide shots
            if abs(ball_pos[1]) > 0.025 and actions[self.keeper_agent_id] in [3, 7]:
                rewards[self.keeper_agent_id] += 0.1

            # 🔁 Encourage varied reactions to ball Y change
            if self.previous_ball_y is not None:
                if abs(ball_pos[1] - self.previous_ball_y) > 0.01 and \
                   actions[self.keeper_agent_id] != self.previous_keeper_action:
                    rewards[self.keeper_agent_id] += 0.05
            self.previous_ball_y = ball_pos[1]
            self.previous_keeper_action = actions[self.keeper_agent_id]

                # ---------------- SHOOTER REWARD SHAPING ----------------

        # 1️⃣ Reward kicking the ball
        if not self.ball_kicked:
            if actions[self.shooter_agent_id] == 12:
                rewards[self.shooter_agent_id] += 0.01
            else:
                rewards[self.shooter_agent_id] += -0.05

        # 2️⃣ After kick: encourage diverse shot direction
        if self.ball_kicked and ball_pos[0] > 1.0:
            if abs(ball_pos[1]) <= 0.02:
                # Center shot — discourage
                rewards[self.shooter_agent_id] += -0.1
            else:
                # Left or right shot — encourage
                rewards[self.shooter_agent_id] += 0.1

        return obs, rewards, done, info

    def reset(self, **kwargs):
        self.ball_kicked = False
        self.goal_reward_given = False
        self.steps_after_kick = 0
        self.y_reward_given = False
        self.previous_ball_pos = None
        self.previous_keeper_pos = None
        self.previous_ball_y = None
        self.previous_keeper_action = None

        obs = self.env.reset(**kwargs)

        # Prevent sticky keeper movement on reset
        release_action = [0, 14]
        obs, _, _, _ = self.env.step(release_action)

        return obs

    def _extract_ball_pos(self, obs):
        return np.array([obs[88], obs[89]])

    def _extract_keeper_pos(self, obs):
        return np.array([obs[44], obs[45]])
