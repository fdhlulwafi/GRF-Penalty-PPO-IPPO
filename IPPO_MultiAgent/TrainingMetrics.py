from stable_baselines3.common.callbacks import BaseCallback

class TrainingMetricsCallback(BaseCallback):
    def __init__(self, verbose=1):
        super().__init__(verbose)

    def _on_step(self) -> bool:
        # Nothing special every step
        return True

    def _on_rollout_end(self) -> None:
        # Flush the logger so metrics get updated
        self.model.logger.dump(self.num_timesteps)

        logger = self.model.logger
        def get_val(name):
            return logger.name_to_value.get(name, float('nan'))

        explained_var = get_val("rollout/explained_variance")
        policy_loss = get_val("train/policy_gradient_loss")
        entropy_loss = get_val("train/entropy_loss")
        value_loss = get_val("train/value_loss")
        clip_frac = get_val("train/clip_fraction")

        print(f"\n🔎 Metrics after rollout:")
        print(f"  Explained Variance: {explained_var:.4f}")
        print(f"  Policy Loss:        {policy_loss:.4f}")
        print(f"  Entropy Loss:       {entropy_loss:.4f}")
        print(f"  Value Loss:         {value_loss:.4f}")
        print(f"  Clip Fraction:      {clip_frac:.4f}")