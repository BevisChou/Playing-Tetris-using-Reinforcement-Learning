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
            # "board_property": gym.spaces.Box(low=0, high=255, shape=(4,), dtype=np.uint8),
            "next_piece": gym.spaces.Discrete(len(self.tetris.pieces))
        })


    def reset(self, seed=0) -> np.ndarray:
        self.last_property = self.tetris.reset()
        obs = {
            "board": self.tetris.get_mask(),
            # "board_property": property.numpy().astype(np.uint8),
            "next_piece": self.tetris.ind
        }
        return obs, {}
        

    def step(self, action: Tuple[int, int]) -> Tuple[np.ndarray, float, bool, dict]:        
        x, num_rotations = action

        # Ensure the piece's horizontal position does not exceed the maximum valid position
        x = min(x, self.tetris.width - len(self.tetris.piece[0]))
        
        _, done, obs, info = self.tetris.step((x, num_rotations))
        
        reward = 5.0 + 1000 * info["lines_cleared"]
        if info["height"] > self.last_property["height"]:
            reward -=  5 * (info["height"] - self.last_property["height"])
        if info["holes"] < self.last_property["holes"]:
            reward += 10 * (self.last_property["holes"] - info["holes"])
        
        self.last_property = info
        
        return obs, reward, done, False, {}


    def render(self, mode="human"):
        self.tetris.render()


    def close(self):
        pass
