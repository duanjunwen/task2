import math
import torch
import sys
import os
import time
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
from torch import nn
import torch.distributed as dist
import initialize as fs_init
from layers import (
    ColumnParallelLinear,
    RowParallelLinear,
)

from utils import split_tensor_along_last_dim

# dist env
def init_dist():
    rank = int(os.environ['RANK'])  # 当前进程的全局排名（Global Rank）
    local_rank = int(os.environ['LOCAL_RANK'])  # 表示当前进程在本地节点中的排名（Local Rank）。
    # single node GPU num :LOCAL RANK node0:{LOCAL_RANK0-3}, node1:{LOCAL_RANK4-7}
    world_size = int(os.environ['WORLD_SIZE'])  # 表示分布式训练中总共的进程数。

    dist.init_process_group(world_size=world_size, rank=rank,
                            init_method="env://", backend="nccl")
    fs_init.initialize_model_parallel(world_size)
    torch.cuda.set_device(local_rank)

# time cycle
def get_time():
    torch.cuda.synchronize()
    return time.time()

def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """torch.repeat_interleave(x, dim=2, repeats=n_rep)"""
    bs, slen, n_kv_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    return (
        x[:, :, :, None, :]
        .expand(bs, slen, n_kv_heads, n_rep, head_dim)
        .reshape(bs, slen, n_kv_heads * n_rep, head_dim)
    )


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
        
        # weight[hidden_dim, dk] : 512, 64
        # wo [hidden_dim, hidden_dim]: 512, 512
        self.wq = torch.randn(self.hidden_dim, self.dk)
        self.wk = torch.randn(self.hidden_dim, self.dk)
        self.wv = torch.randn(self.hidden_dim, self.dk)
        self.wo = torch.randn(self.dk, self.hidden_dim)
        
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
class Attention_TP(nn.Module):
    """
    compute scale dot product attention

    Query : given sentence that we focused on (decoder)
    Key : every sentence to check relationship with Qeury(encoder)
    Value : every sentence same with Key (encoder)
    """

    def __init__(self, batch_size, head, seq_length, hidden_dim):
        super().__init__()
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


def main():
    # print(f"CUDA device {int(os.environ['LOCAL_RANK'])}")
    device = int(os.environ['LOCAL_RANK'])
    batch_size, head, seq_length, dim = 64, 8, 128, 4096
    # atten no TP
    attn = Attention(batch_size, head, seq_length, dim)
    dk = dim // head # 512 / 8 = 64
    x = torch.randn(batch_size, head, seq_length, dim)
    # q, xk, v = torch.randn(batch_size, head, seq_length, dk), torch.randn(batch_size, head, seq_length, dk), torch.randn(batch_size, head, seq_length, dk)
    AttenStart = get_time()
    score = attn.forward(x)
    AttenStop = get_time()
    
    # atten with CP (columnParallism)
    init_dist()
    x_tp = x.cuda(device=device)
    attn_cp = Attention_TP(batch_size, head, seq_length, dim)
    # if attn_cp.supports_gradient_checkpointing:
    #     attn_cp.gradient_checkpointing_enable()
    #     print(f"Gradient Checkpointing: {attn_cp.is_gradient_checkpointing}")
    dk = dim // head # 512 / 8 = 64
    # q, xk, v = torch.randn(batch_size, head, seq_length, dk), torch.randn(batch_size, head, seq_length, dk), torch.randn(batch_size, head, seq_length, dk)
    AttenCPStart = get_time()
    score = attn_cp.forward(x_tp)
    AttenCPStop = get_time()
    
    # # atten with RP(rowParallism)
    # # pass
    # attn_rp = ScaleDotProductAttention_RP(batch_size, head, seq_length, dim)
    # AttenRPStart = get_time()
    # score, v = attn_rp.forward(x)
    # AttenRPStop = get_time()
    
    print(f" Atten (No TP) {AttenStop - AttenStart}; Atten (with CP) {AttenCPStop - AttenCPStart}; Atten (with RP) {None}")
    


if __name__ == "__main__":
    main()
    
    
    