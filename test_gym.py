from src.tetris_gym import TetrisEnv
from stable_baselines3 import A2C

env = TetrisEnv()

model = A2C.load("models/a2c", env=env)

obs, _ = env.reset()
done = False
while not done:
    action, _state = model.predict(obs, deterministic=True)
    obs, reward, done, truncated, info = env.step(action)