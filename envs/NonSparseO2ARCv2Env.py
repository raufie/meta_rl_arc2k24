from arcle.envs import O2ARCv2Env
from typing import SupportsFloat
import numpy as np 

class NonSparseO2ARCv2Env(O2ARCv2Env):
    def __init__(self,Config, **kwargs):
        super().__init__(**kwargs)
        self.Config = Config
        
    def _grid_accuracy(self, grid, answer):
        return (grid==answer).mean()

    def _dim_mse_complement(self, grid_dim, answer_dim):
        mse =  (((grid_dim - answer_dim)/29)**2).sum()/2
        return 1 - mse

    def reward(self, state)-> SupportsFloat:
        
        h,w = self.answer.shape

# ACCURATE SUBMIT BOOST
        submit_boost = 0
        if self.last_action_op == len(self.operations)-1:

            if tuple(state['grid_dim']) == self.answer.shape:               
                if np.all(state['grid'][0:h, 0:w] == self.answer):
                    submit_boost = self.Config.REWARD_CORRECT_BOOST
# Grid reward 
        grid_acc = self._grid_accuracy(state['grid'][0:h, 0:w], self.answer)
        dim_mse_complement = self._dim_mse_complement(state['grid_dim'], self.answer.shape)

        return float(grid_acc*self.Config.REWARD_GRID_WEIGHT + dim_mse_complement*self.Config.REWARD_DIM_WEIGHT + submit_boost*self.Config.REWARD_SUBMIT_WEIGHT)