import gym
import matplotlib.pyplot as plt
import numpy as np
from ray.train.rl.rl_predictor import RLPredictor
from ray.air import Checkpoint
import matplotlib.pyplot as plt

checkpoint = Checkpoint.from_directory("/home/ray/ray_results/AIRPPO_2023-02-16_05-31-56/AIRPPO_4acc9_00000_0_2023-02-16_05-32-01/checkpoint_000100")



num_episodes = 50
predictor = RLPredictor.from_checkpoint(checkpoint)



env = gym.make("CartPole-v1")
r = [0]
rw = 0
eps = [0]
for i in range(num_episodes):
    obs = env.reset()
    done = False
    while not done:
        action = predictor.predict(np.array([obs]))
        action = int(action)
        obs, reward, done, info = env.step(action)
        #env.render()
        rw += reward
    r.append(rw)
    eps.append(i)
    rw = 0

print(r)

plt.xlabel("episodes")
plt.ylabel("reward")
plt.plot(eps,r)
plt.show()





"""
predictor = RLPredictor().from_checkpoint("ray/home/ray_results/PPO/PPO_StatelessCartPole_85b0b_00000_0_2023-02-16_01-23-07/checkpoint_000250")
num_episodes = 50
env = gym.make("CartPole-v1")
rewards = 0
for i in range(num_episodes):
    obs = env.reset()
    reward = 0.0
    done = False
    while not done:
        env.render()
        action = predictor.predict(np.array([obs]))
        obs, r, done, _ = env.step(action[0])
        rewards += r
    print(rewards)

ray.init(num_cpus=8,num_gpus=1)
"""