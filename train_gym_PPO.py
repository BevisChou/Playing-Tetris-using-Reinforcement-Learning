from src.tetris_gym import TetrisEnv
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.env_util import make_vec_env

env = TetrisEnv()
check_env(env)

vec_env = make_vec_env(TetrisEnv, n_envs=30)

from stable_baselines3 import PPO
total_timesteps = 300000
model = PPO("CnnPolicy", env, verbose=1, tensorboard_log="./log/ppo_cnnPolicy")
model.learn(total_timesteps=total_timesteps, log_interval=4, tb_log_name="first_run")
model.save("models/ppo_{}".format(total_timesteps))
