import argparse
import os
import shutil
from random import random, randint, sample

import numpy as np
import torch
import torch.nn as nn
from tensorboardX import SummaryWriter

from src.deep_q_network import DeepQNetwork
from src.tetris import Tetris
from collections import deque


def train(opt):
    # 设置DVL的log存储路径，可视化训练过程
    if os.path.isdir(opt.log_path):
        shutil.rmtree(opt.log_path)
    os.makedirs(opt.log_path)

    # 实例化环境,俄罗斯方块游戏的宽与高,方块大小
    env = Tetris(width=opt.width, height=opt.height, block_size=opt.block_size)

    # 实例化网络,这个网络是整个实验的核心,在训练中输出动作,指导游戏进行
    model = DeepQNetwork()

    # 设置学习率,优化器等
    lr = opt.lr
    optimizer = paddle.optimizer.Adam(parameters=model.parameters(), learning_rate=lr)
    criterion = nn.MSELoss()

    # 这里初始化游戏环境,相当于新建一局游戏
    state = env.reset()

    # 这个replay_memory是关键点,相当于记忆库,提供给网络进行学习
    # 记忆库的设计非常重要,要有一定的随机动作来走出局部最优解,也要有一些高得分的动作加速网络收敛
    # replay_memory_size是设置记忆库的大小,个人感觉大一点好
    replay_memory = deque(maxlen=opt.replay_memory_size)
    epoch = 0
    with LogWriter(logdir=opt.log_path) as writer:
        while epoch < opt.num_epochs:
            # 开始训练,env.get_next_states(),游戏就会进入下一步,在游戏中表现为在最上方给出一个方块
            next_steps = env.get_next_states()
            # Exploration or exploitation,这是文章的第二个关键点,控制replay_memory中随机动作与网络输出动作的比值
            # 一开始由于网络随机初始化不具备能力,随机动作较多,随着训练网络能力增强,随机动作减少,但一直保持一定的比例,来避免陷入局部最优无法跳出
            epsilon = opt.final_epsilon + (max(opt.num_decay_epochs - epoch, 0) * (
                    opt.initial_epsilon - opt.final_epsilon) / opt.num_decay_epochs)
            u = random()
            random_action = u <= epsilon

            # 打包动作与状态
            next_actions, next_states = zip(*next_steps.items())
            next_states = paddle.stack(next_states)

            # 执行随机动作,或执行网络输出的动作
            if random_action:
                index = randint(0, len(next_steps) - 1)
            else:
                model.eval()
                with paddle.no_grad():
                    predictions = model(next_states)[:, 0]
                index = paddle.argmax(predictions).item()

            # 执行动作,获取下一个状态,获取reward,获取游戏是否结束
            next_state = next_states[index, :]
            action = next_actions[index]
            reward, done = env.step(action, render=True)

            # 至此,获取了强化学习所需的所有要素[state, reward, next_state, done],放入replay_memory中,作为训练材料备用
            replay_memory.append([state, reward, next_state, done])
            if done:
                final_score = env.score
                final_tetrominoes = env.tetrominoes
                final_cleared_lines = env.cleared_lines
                state = env.reset()
            else:
                state = next_state
                continue
            # 这一句是判断replay_memory中的素材是否足够,等足够再开始训练,避免数据量太小无法挖掘出可用的知识
            if len(replay_memory) < opt.replay_memory_size / 10:
                print(len(replay_memory) / (opt.replay_memory_size / 10))
                continue
            # 开始训练
            model.train()
            epoch += 1
            # 采样,转张量
            batch = sample(replay_memory, min(len(replay_memory), opt.batch_size))
            state_batch, reward_batch, next_state_batch, done_batch = zip(*batch)
            state_batch = paddle.stack(tuple(state for state in state_batch))
            reward_batch = paddle.to_tensor(np.array(reward_batch, dtype=np.float32)[:, None])
            next_state_batch = paddle.stack(tuple(state for state in next_state_batch))

            # 状态state_batch进入网络,估算q_values
            q_values = model(state_batch)

            # 预测下一个状态
            model.eval()
            with paddle.no_grad():
                next_prediction_batch = model(next_state_batch)
            model.train()

            # 理解了这一句,就理解了整个DQN算法,这一段在下一个Markdown里面详解
            y_batch = paddle.concat(
                tuple(reward if done else reward + opt.gamma * prediction for reward, done, prediction in
                      zip(reward_batch, done_batch, next_prediction_batch)))[:, None]

            # 计算损失,梯度回传,完成一步训练
            optimizer.clear_grad()
            loss = criterion(q_values, y_batch)
            loss.backward()
            optimizer.step()

            # 打印指标
            print("Epoch: {}/{}, Action: {}, Score: {}, Tetrominoes {}, Cleared lines: {}".format(
                epoch,
                opt.num_epochs,
                action,
                final_score,
                final_tetrominoes,
                final_cleared_lines))

            # 写如log日志
            writer.add_scalar(tag="Train/Score", step=epoch - 1, value=final_score)
            writer.add_scalar(tag="Train/Tetrominoes", step=epoch - 1, value=final_tetrominoes)
            writer.add_scalar(tag="Train/Cleared lines", step=epoch - 1, value=final_cleared_lines)

            if epoch > 0 and epoch % opt.save_interval == 0:
                paddle.save(model.state_dict(), "{}/tetris_{}".format(opt.saved_path, epoch))

    paddle.save(model.state_dict(), "{}/tetris".format(opt.saved_path))