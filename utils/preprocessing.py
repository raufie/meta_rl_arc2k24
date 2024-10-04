import numpy as np
import torch
import yaml

class Parser:
    @staticmethod
    def parse_bbox(x, y, w, h):
        # Domain(x,y,w,h) = [0,1]    
        # scale it up
        x = int(np.clip(x, 0,1)*30)
        y = int(np.clip(y, 0,1)*30)
        w = int(np.clip(w, 0,1)*30)
        h = int(np.clip(h, 0,1)*30)

        mask = np.zeros(shape=(30,30), dtype=int)
        mask[y:y+h, x:x+w] = 1
        
        return mask

    def parse_input(obs, action):
        pass
        # action, rewards are for previous timestep

def get_adder(num = 2071, min_dim = 128, n_heads = 8):
    rem = (num+128)%n_heads
    return num + min_dim + n_heads - rem

def parse_inputs(grid, answer, bbox, op, reward, Config):
    grid_tensor = torch.as_tensor(grid).unsqueeze(0).unsqueeze(0).to(Config.DEVICE)
    answer_tensor = torch.as_tensor(answer).unsqueeze(0).unsqueeze(0).to(Config.DEVICE)
    bbox_tensor = torch.as_tensor(Parser.parse_bbox(*bbox)).unsqueeze(0).unsqueeze(0).to(Config.DEVICE)
    op_tensor = torch.as_tensor(op).view(1).to(Config.DEVICE)
    reward_tensor = torch.tensor(reward).view(1,1).to(Config.DEVICE)
    return [grid_tensor, answer_tensor, bbox_tensor, op_tensor, reward_tensor]