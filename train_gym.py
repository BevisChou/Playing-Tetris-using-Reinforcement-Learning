from src.tetris_gym import TetrisEnv
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.env_util import make_vec_env

env = TetrisEnv()
check_env(env)
# env = make_vec_env(TetrisEnv, n_envs=30)

from stable_baselines3 import A2C

load = False # Please set this accordingly

model = None
if load:
    model = A2C.load("models/a2c", env=env, verbose=1, tensorboard_log="./log/a2c")
else:
    model = A2C("MultiInputPolicy", env=env, verbose=1, tensorboard_log="./log/a2c")

model.learn(total_timesteps=500_000, tb_log_name="a2c", reset_num_timesteps=not load)
model.save("models/a2c")