import math
import torch
import sys
import os
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
from torch import nn
import torch.optim as optim
import torch.distributed as dist
import initialize as fs_init
from layers import (
    ColumnParallelLinear,
    RowParallelLinear,
)
from torch.utils.checkpoint import checkpoint as ckpt

# self-Attention
class Attention(nn.Module):
    """
    compute scale dot product attention

    Query : given sentence that we focused on (decoder)
    Key : every sentence to check relationship with Qeury(encoder)
    Value : every sentence same with Key (encoder)
    """

    def __init__(self, batch_size, head, seq_length, hidden_dim):
        super(Attention, self).__init__()
        self.batch_size = batch_size
        self.head = head
        self.seq_length = seq_length
        self.hidden_dim = hidden_dim
        self.dk = self.hidden_dim // self.head
        self.device = int(os.environ['LOCAL_RANK'])
        
        # weight[hidden_dim, dk] : 512, 64
        # wo [hidden_dim, hidden_dim]: 512, 512
        self.wq = torch.randn(self.hidden_dim, self.dk).to(self.device)
        self.wk = torch.randn(self.hidden_dim, self.dk).to(self.device)
        self.wv = torch.randn(self.hidden_dim, self.dk).to(self.device)
        self.wo = torch.randn(self.dk, self.hidden_dim).to(self.device)
        
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None, e=1e-12):
        # input is 4 dimension tensor
        # 1. dot product Query with Key^T to compute similarity
        q, k, v = x @ self.wq, x @ self.wk, x @ self.wv
        k_t = k.transpose(2, 3)  # transpose
        score = (q @ k_t) / math.sqrt(self.hidden_dim)  # scaled dot product

        # 2. apply masking (opt)
        if mask is not None:
            score = score.masked_fill(mask == 0, -10000)

        # 3. pass them softmax to make [0, 1] range
        score = self.softmax(score)

        # 4. multiply with Value
        v = score @ v
        
        score = v @ self.wo
        
        return score
        

        # return v, score


# self-Attention ColumnParalism
class AttentionTP(nn.Module):
    """
    compute scale dot product attention

    Query : given sentence that we focused on (decoder)
    Key : every sentence to check relationship with Qeury(encoder)
    Value : every sentence same with Key (encoder)
    """

    def __init__(self, batch_size, head, seq_length, hidden_dim):
        super(AttentionTP,  self).__init__()
        self.batch_size = batch_size
        self.head = head
        self.seq_length = seq_length
        self.hidden_dim = hidden_dim
        self.dk = self.hidden_dim // self.head
        
        model_parallel_size = fs_init.get_model_parallel_world_size()
        self.local_head = self.head // model_parallel_size
        self.head_dim = self.hidden_dim // self.head
        
        # weight[hidden_dim, dk] : 512, 64
        self.wq = ColumnParallelLinear(
            self.hidden_dim,
            self.dk,
            bias=False,
            gather_output=False,
            init_method=lambda x: x,
            
        )
        self.wk = ColumnParallelLinear(
            self.hidden_dim,
            self.dk,
            bias=False,
            gather_output=False,
            init_method=lambda x: x,
        )
        self.wv = ColumnParallelLinear(
            self.hidden_dim,
            self.dk,
            bias=False,
            gather_output=False,
            init_method=lambda x: x,
        )
        
        self.wo = RowParallelLinear(
            self.hidden_dim,
            self.hidden_dim,
            bias=False,
            input_is_parallel=True,
            init_method=lambda x: x,
        )
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None, e=1e-12):
        # input is 4 dimension tensor
        # 1. dot product Query with Key^T to compute similarity
        # x[64, 8, 128, 512] @ wq [512, 64]
        # q, k, v [64, 8, 128, 64]
        
        # Parallism
        # Col Para切分wq, wk, wv last dim wq[512,8]
        # x[64, 8, 128, 512] @ wq [512, 8] = q[64, 8, 128, 8] (k, v same shape)
        
        xq = self.wq.forward(x)  
        xk = self.wk.forward(x)
        xv = self.wv.forward(x)
        
        
        # kt[64, 8, 8, 128] = xk[64, 8, 128, 8] tranpose last 2 dim
        k_t = xk.transpose(2, 3)  
        
        # xq [64, 8, 128, 8] @ kt[64, 8, 8, 128] = score [64, 8, 128, 128]
        score = (xq @ k_t) / math.sqrt(self.hidden_dim)  # scaled dot product

        # 2. apply masking (opt)
        if mask is not None:
            score = score.masked_fill(mask == 0, -10000)

        # 3. pass them softmax to make [0, 1] range
        # score [64, 8, 128, 128]
        score = self.softmax(score).type_as(xq)

        # 4. multiply with Value
        # score [64, 8, 128, 128] @ xv [64, 8, 128, 8] = output[64, 8, 128, 8]
        output = score @ xv
        
        # output[64, 8, 128, 8] -transpose(1,2)-> output[64, 128, 8, 8] --> output[64, 128, 64]
        output = output.transpose(1, 2).contiguous().view(self.batch_size, self.seq_length, -1)
        output = self.wo.forward(output)
        
        # 5.output[64, 128, 64] * wo splite[64, 512] = output[64, 128, 512]
        # all reduce output[64, 128, 512]
        
        # print(f"x dim {x.shape}, kt shape {k_t.shape}, xq shape {xq.shape}, score shape {score.shape}, xv shape {xv.shape}, output shape {output.shape}\n")
        
        return output


        # return v, score


# self-Attention ColumnParalism
class AttentionTpCkpt(nn.Module):
    """
    compute scale dot product attention

    Query : given sentence that we focused on (decoder)
    Key : every sentence to check relationship with Qeury(encoder)
    Value : every sentence same with Key (encoder)
    """

    def __init__(self, batch_size, head, seq_length, hidden_dim):
        super(AttentionTpCkpt,  self).__init__()
        self.batch_size = batch_size
        self.head = head
        self.seq_length = seq_length
        self.hidden_dim = hidden_dim
        self.dk = self.hidden_dim // self.head
        
        model_parallel_size = fs_init.get_model_parallel_world_size()
        self.local_head = self.head // model_parallel_size
        self.head_dim = self.hidden_dim // self.head
        
        # weight[hidden_dim, dk] : 512, 64
        self.wq = ColumnParallelLinear(
            self.hidden_dim,
            self.dk,
            bias=False,
            gather_output=False,
            init_method=lambda x: x,
            
        )
        self.wk = ColumnParallelLinear(
            self.hidden_dim,
            self.dk,
            bias=False,
            gather_output=False,
            init_method=lambda x: x,
        )
        self.wv = ColumnParallelLinear(
            self.hidden_dim,
            self.dk,
            bias=False,
            gather_output=False,
            init_method=lambda x: x,
        )
        
        self.wo = RowParallelLinear(
            self.hidden_dim,
            self.hidden_dim,
            bias=False,
            input_is_parallel=True,
            init_method=lambda x: x,
        )
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None, e=1e-12):
        # input is 4 dimension tensor
        # 1. dot product Query with Key^T to compute similarity
        # x[64, 8, 128, 512] @ wq [512, 64]
        # q, k, v [64, 8, 128, 64]
        
        # Parallism
        # Col Para切分wq, wk, wv last dim wq[512,8]
        # x[64, 8, 128, 512] @ wq [512, 8] = q[64, 8, 128, 8] (k, v same shape)
        
        # xq = self.wq.forward(x)  
        # xk = self.wk.forward(x)
        # xv = self.wv.forward(x)
        
        xq = ckpt(self._xq_checkpoint_forward, x)
        xk = ckpt(self._xk_checkpoint_forward, x)
        xv = ckpt(self._xv_checkpoint_forward, x)
        
        
        # kt[64, 8, 8, 128] = xk[64, 8, 128, 8] tranpose last 2 dim
        k_t = xk.transpose(2, 3)  
        
        # xq [64, 8, 128, 8] @ kt[64, 8, 8, 128] = score [64, 8, 128, 128]
        score = (xq @ k_t) / math.sqrt(self.hidden_dim)  # scaled dot product

        # 2. apply masking (opt)
        if mask is not None:
            score = score.masked_fill(mask == 0, -10000)

        # 3. pass them softmax to make [0, 1] range
        # score [64, 8, 128, 128]
        score = self.softmax(score).type_as(xq)

        # 4. multiply with Value
        # score [64, 8, 128, 128] @ xv [64, 8, 128, 8] = output[64, 8, 128, 8]
        output = score @ xv
        
        # output[64, 8, 128, 8] -transpose(1,2)-> output[64, 128, 8, 8] --> output[64, 128, 64]
        output = output.transpose(1, 2).contiguous().view(self.batch_size, self.seq_length, -1)
        output = self.wo.forward(output)
        
        # 5.output[64, 128, 64] * wo splite[64, 512] = output[64, 128, 512]
        # all reduce output[64, 128, 512]
        
        # print(f"x dim {x.shape}, kt shape {k_t.shape}, xq shape {xq.shape}, score shape {score.shape}, xv shape {xv.shape}, output shape {output.shape}\n")
        
        return output


    def _xq_checkpoint_forward(self, x):
        xq = self.wq.forward(x)  
        return xq
    
    def _xk_checkpoint_forward(self, x):
        xk = self.wk.forward(x)  
        return xk
    
    def _xv_checkpoint_forward(self, x):
        xv = self.wv.forward(x)  
        return xv