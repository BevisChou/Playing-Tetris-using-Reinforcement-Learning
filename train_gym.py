from src.tetris_gym import TetrisEnv
from stable_baselines3.common.env_checker import check_env

env = TetrisEnv()
check_env(env)

from stable_baselines3 import A2C

model = A2C("MultiInputPolicy", env, verbose=1, tensorboard_log="./log/a2c")
model.learn(total_timesteps=1_000, tb_log_name="first_run")