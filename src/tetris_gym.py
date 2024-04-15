import gymnasium as gym
import numpy as np

from src.tetris import Tetris

from typing import Tuple

class TetrisEnv(gym.Env):
    def __init__(self, height=20, width=10, block_size=30):
        super().__init__()
        self.tetris = Tetris(height, width, block_size)
        self.action_space = gym.spaces.MultiDiscrete([width, 4])
        self.observation_space = gym.spaces.Dict({
            "board": gym.spaces.Box(low=0, high=1, shape=(height, width), dtype=np.uint8),
            "next_piece": gym.spaces.Discrete(len(self.tetris.pieces))
        })


    def reset(self, seed=0) -> np.ndarray:
        self.tetris.reset()
        return self.__get_state(), {}
        

    def step(self, action: Tuple[int, int]) -> Tuple[np.ndarray, float, bool, dict]:        
        x, num_rotations = action

        # Ensure the piece's horizontal position does not exceed the maximum valid position
        x = min(x, self.tetris.width - len(self.tetris.piece[0]))
        
        reward, done = self.tetris.step((x, num_rotations))
        
        return self.__get_state(), reward, done, False, {}


    def render(self, mode="human"):
        self.tetris.render()


    def close(self):
        pass


    def __get_state(self):
        return {
            "board": self.tetris.get_mask(),
            "next_piece": self.tetris.ind
        }
