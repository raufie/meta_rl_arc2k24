import torch
import torch.nn as nn
import torch.nn.functional as F

class Config:
    GRID_KERNEL_SIZE = 3
    GRID_PROJ_KERNEL_SIZE = 3
    BBOX_KERNEL_SIZE = 3

# ops
    OPERATION_EMB_DIM = 131
    N_OPS = 36
    NO_OP = 35 # 35th ix = 36th action

# full state embedding
    EMB_DIM = 2701 + 128 + 8 - 5
    # please see the function get_adder in utils for how I got this number
    BLOCK_SIZE = 64

# Decoder config
    N_HEADS = 8
    RESIDUAL_DROP = 0.2
    N_BLOCKS = 8
# outs
    # output op_size (NO_OP not allowed)
    OP_SIZE = 34

# DEVICE
    DEVICE = 'cpu'

class DecoderBlock(nn.Module):
    def __init__(self, CONFIG):
        super(DecoderBlock, self).__init__()

        self.ln1 = nn.LayerNorm(CONFIG.EMB_DIM)

        self.mha = nn.MultiheadAttention(embed_dim=CONFIG.EMB_DIM, num_heads=CONFIG.N_HEADS, batch_first=True)
        
        self.ffn = nn.Sequential(
            nn.Linear(CONFIG.EMB_DIM, CONFIG.EMB_DIM*4),
            nn.GELU(),
            nn.Linear(CONFIG.EMB_DIM*4,CONFIG.EMB_DIM),
            nn.Dropout(CONFIG.RESIDUAL_DROP)
        )
        self.ln2 = nn.LayerNorm(CONFIG.EMB_DIM)

    def forward(self, x, padding_mask=None):
        B, T, C = x.shape

        # Apply LayerNorm
        x_norm = self.ln1(x)

                
        causal_mask = torch.triu(torch.ones(T, T), diagonal=1).bool().to(x.device)
        causal_mask = causal_mask.float().masked_fill(causal_mask, float('-inf')).to(x.device)
        
        # Multi-head attention
        mha_output, _ = self.mha(x_norm, x_norm, x_norm, attn_mask=causal_mask, key_padding_mask=padding_mask)
        
        # Residual connection and FeedForward layer
        x = x + mha_output
        x = x + self.ffn(self.ln2(x))
        
        return x

class GridEncoder(nn.Module):
    """
    Takes a T,1,H,W tensor and returns a learned projections of the form T,H*W
    """
    def __init__(self,convolutional_kernel_size=3, convolutional_projection_kernel_size=3):
        super(GridEncoder, self).__init__()

        self.convolution = nn.Conv2d(11,1, kernel_size=convolutional_kernel_size, padding='same')
        self.projection = nn.Conv2d(1,1, kernel_size=convolutional_projection_kernel_size, padding='same')
        self.flatten = nn.Flatten(2,3)

    def forward(self, raw_grid):
        processed_grid = self._pre_process(raw_grid)

        
        out = self.convolution(processed_grid)
        out = self.projection(out)
        out = self.flatten(out).squeeze(1)


        return out

    def _pre_process(self, raw_grid: torch.Tensor):
        # 10 bin mask encodings, 1 normalized full encoding
        # REturns B,T,C,H,W.. where C=11
        assert isinstance(raw_grid,torch.Tensor) 
        assert raw_grid.ndim == 4, "expect [T,C,H,W]"

        # Batch will be 1 for now (my gpu prolly wont be able to handle many anyway)
        # T is the actual batch dim for conv2D

        # T,C,H,W
        masks = []
        for i in range(10):
            masks.append((raw_grid == i).int())
        grid_processed = torch.cat([*masks, raw_grid/9], dim=1)
    
        return grid_processed

class BBoxEncoder(nn.Module):
    """
    Takes a T,1,H,W tensor 1and returns a learned projections of the form T,H*W
    """
    def __init__(self, convolutional_projection_kernel_size=3):
        super(BBoxEncoder, self).__init__()
        self.projection = nn.Conv2d(1,1, kernel_size=convolutional_projection_kernel_size, padding='same')
        self.flatten = nn.Flatten(2,3)

    def forward(self, raw_grid):       
        
        out = self.projection(raw_grid)
        out = self.flatten(out).squeeze(1)

        return out


class Mark_I_DTPolicy(nn.Module):
# idx=34 (no_op (before first op) (unchoosable later))
    def __init__(self, Config: Config):
        super(Mark_I_DTPolicy, self).__init__()
        
        # INPUT ENCODING
        self.answer_encoder = GridEncoder(convolutional_kernel_size = Config.GRID_KERNEL_SIZE, convolutional_projection_kernel_size = Config.GRID_PROJ_KERNEL_SIZE)
        self.grid_encoder = GridEncoder(convolutional_kernel_size = Config.GRID_KERNEL_SIZE, convolutional_projection_kernel_size = Config.GRID_PROJ_KERNEL_SIZE)
        self.bbox_proj =  GridEncoder(convolutional_projection_kernel_size = Config.BBOX_KERNEL_SIZE)

        #   Operation Embedding
        self.operation_embedding = nn.Embedding(Config.N_OPS, Config.OPERATION_EMB_DIM)
        self.global_position_embedding = nn.Embedding(Config.BLOCK_SIZE, Config.EMB_DIM)

        # Global Linear Projection
        self.global_linear_projection = nn.Linear(Config.EMB_DIM, Config.EMB_DIM )

        self.blocks = nn.ModuleList([DecoderBlock(Config) for _ in range(Config.N_BLOCKS)])

        self.prehead_ffn = nn.Sequential(
            nn.Linear(Config.EMB_DIM, Config.EMB_DIM),
            nn.GELU(),
            nn.Linear(Config.EMB_DIM,Config.EMB_DIM),
            nn.Dropout(Config.RESIDUAL_DROP)
        )

        self.ln = nn.LayerNorm(Config.EMB_DIM)
        self.op_head = nn.Linear(Config.EMB_DIM, Config.OP_SIZE)
        self.bbox_ln = nn.Linear(Config.EMB_DIM, 4)
        self.bbox_act = nn.Sigmoid() 

    def forward(self, grid: torch.Tensor, answer: torch.Tensor, bbox_mask: torch.Tensor, operation:torch.Tensor = None, reward:torch.Tensor=None, **kwargs):
        """
        current information (t i.e St )

        grid: T,30,30
        answer: T,30,30
        
        # past context (t-1 i.e at-1, rt-1)

        bbox_mask: T,30,30

        operation: T,1 (int)
        reward:T,1 (float)
        """
        T_IDX = torch.arange(grid.shape[0]).to(grid.device)

        grid_enc = self.grid_encoder(grid)
        answer_enc = self.answer_encoder(answer)
        
        # previous action and reward encodings
        bbox_projection = self.bbox_proj(bbox_mask)
        op_emb = self.operation_embedding(operation)

        positional_encoding = self.global_position_embedding(T_IDX)
        state_encoding = torch.cat([op_emb, reward, bbox_projection, answer_enc, grid_enc], dim=-1)
        pre_emb = state_encoding + positional_encoding

        # linear projection + add batch
        x = self.global_linear_projection(pre_emb).unsqueeze(0)                 
        
        for block in self.blocks:
            x = block(x)
        x = self.ln(x)
        op_logits = self.op_head(x)

        bbox = self.bbox_act(self.bbox_ln(x))
        x = torch.softmax(x, dim=-1)
        return op_logits, bbox
