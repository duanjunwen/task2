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

# self-Attention
class ScaleDotProductAttention(nn.Module):
    """
    compute scale dot product attention

    Query : given sentence that we focused on (decoder)
    Key : every sentence to check relationship with Qeury(encoder)
    Value : every sentence same with Key (encoder)
    """

    def __init__(self, batch_size, head, seq_length, hidden_dim):
        super(ScaleDotProductAttention, self).__init__()
        self.batch_size = batch_size
        self.head = head
        self.seq_length = seq_length
        self.hidden_dim = hidden_dim
        self.dk = self.hidden_dim // self.head
        
        # weight[hidden_dim, dk] : 512, 64
        self.wq = torch.randn(self.hidden_dim, self.dk)
        self.wk = torch.randn(self.hidden_dim, self.dk)
        self.wv = torch.randn(self.hidden_dim, self.dk)
        
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

        return v, score


# self-Attention
class ScaleDotProductAttention_CP(nn.Module):
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
        
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None, e=1e-12):
        # input is 4 dimension tensor
        # 1. dot product Query with Key^T to compute similarity
        q = self.wq.forward(x)
        k = self.wq.forward(x)
        v = self.wq.forward(x)
        
        k_t = k.transpose(2, 3)  # transpose
        
        score = (q @ k_t) / math.sqrt(self.hidden_dim)  # scaled dot product

        # 2. apply masking (opt)
        if mask is not None:
            score = score.masked_fill(mask == 0, -10000)

        # 3. pass them softmax to make [0, 1] range
        score = self.softmax(score)

        # 4. multiply with Value
        v = score @ v

        return v, score


def main():
    batch_size, head, seq_length, dim = 64, 8, 128, 4096
    # atten no TP
    attn = ScaleDotProductAttention(batch_size, head, seq_length, dim)
    dk = dim // head # 512 / 8 = 64
    x = torch.randn(batch_size, head, seq_length, dim)
    # q, k, v = torch.randn(batch_size, head, seq_length, dk), torch.randn(batch_size, head, seq_length, dk), torch.randn(batch_size, head, seq_length, dk)
    AttenStart = get_time()
    score, v = attn.forward(x)
    AttenStop = get_time()
    
    # atten with CP (columnParallism)
    init_dist()
    attn_tp = ScaleDotProductAttention_CP(batch_size, head, seq_length, dim)
    dk = dim // head # 512 / 8 = 64
    x = torch.randn(batch_size, head, seq_length, dim)
    # q, k, v = torch.randn(batch_size, head, seq_length, dk), torch.randn(batch_size, head, seq_length, dk), torch.randn(batch_size, head, seq_length, dk)
    AttenCPStart = get_time()
    score, v = attn_tp.forward(x)
    AttenCPStop = get_time()
    
    # atten with RP(rowParallism)
    # pass
    AttenRPStart = get_time()
    AttenRPStop = get_time()
    
    print(f"Atten (No TP) {AttenStop - AttenStart}; Atten (with CP) {AttenCPStop - AttenCPStart}; Atten (with RP) {AttenRPStop - AttenRPStart}")
    


if __name__ == "__main__":
    main()
    
    
    