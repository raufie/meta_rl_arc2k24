import torch

class ContextBuffer:
    def __init__(self, block_size):
        # St(Grid_t,Answer,BBox_t_1, OP_t_1, Reward_t_1)
        self.buffer = []
        self.block_size = block_size
    def update(self, parsed_input):
        # 0,1,2 (grids) (T dim = 0)
        # 3 op (no T dim)
        # 4 reward (T dim = 0)
        # TODO: handle T dim = Config.BlockSize

        if len(self.buffer) == 0:
            self.buffer = parsed_input[:]
            return self.buffer

        for i in range(len(self.buffer)):
            self.buffer[i] = torch.cat([self.buffer[i][-self.block_size+1:,...], parsed_input[i]], dim=0)
        return self.buffer
