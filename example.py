import time
import gym
import numpy as np

env = gym.make('FetchReach-v1')
obs = env.reset()
done = False
while not done:
    action = obs['desired_goal'] - obs['achieved_goal']
    action = np.append(action, 0)
    obs, reward, done, info = env.step(action)
    env.render()
time.sleep(5)
env.close()
