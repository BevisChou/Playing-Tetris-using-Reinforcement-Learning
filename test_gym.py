import numpy as np 

from src.tetris_gym import TetrisEnv
from stable_baselines3 import A2C

env = TetrisEnv()

model = A2C.load("models/a2c", env=env)

n_test, reward_sums = 1000, []

for i in range(n_test):
    obs, _ = env.reset()
    reward_sum, done = 0, False
    while not done:
        action, _state = model.predict(obs, deterministic=True)
        obs, reward, done, _, _ = env.step(action)
        reward_sum += reward
    reward_sums.append(reward_sum)
    print(reward_sum)

print(np.mean(reward_sums), np.var(reward_sums))
print(np.percentile(reward_sums, [0, 25, 50, 75, 100]))