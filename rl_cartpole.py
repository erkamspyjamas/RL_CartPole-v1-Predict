from ray.air.config import RunConfig, ScalingConfig
from ray.train.rl import RLTrainer
import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    "--stop-iters", type=int, default=100, help="Number of iterations to train."
)
parser.add_argument(
    "--stop-timesteps", type=int, default=1000000, help="Number of timesteps to train."
)
parser.add_argument(
    "--stop-reward", type=float, default=250000.0, help="Reward at which we stop training."
)
args = parser.parse_args()
stop = {
        "training_iteration": args.stop_iters,
        "timesteps_total": args.stop_timesteps,
        "episode_reward_mean": args.stop_reward,
    }
trainer = RLTrainer(
    run_config=RunConfig(stop=stop),
    scaling_config=ScalingConfig(num_workers=5, use_gpu=True),
    algorithm="PPO",
    config={
        "env": "CartPole-v1",
        "framework": "torch",
        "evaluation_num_workers": 1,
        "evaluation_interval": 1,
        "evaluation_config": {"input": "sampler"},

    },
)
result = trainer.fit()

print(result.checkpoint)