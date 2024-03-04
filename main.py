import math
import torch
import sys
import os
import time
import argparse
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

from utils import split_tensor_along_last_dim

# dist env
def init_dist():
    rank = int(os.environ['RANK'])  # 当前进程的全局排名（Global Rank）
    local_rank = int(os.environ['LOCAL_RANK'])  # 表示当前进程在本地节点中的排名（Local Rank）。
    # single node GPU num :LOCAL RANK node0:{LOCAL_RANK0-3}, node1:{LOCAL_RANK4-7}
    world_size = int(os.environ['WORLD_SIZE'])  # 表示分布式训练中总共的进程数。

    dist.init_process_group(world_size=world_size, rank=rank,
                            init_method="env://", backend="nccl")
    # model_parallel_size_= world_size，GPU全部用于tensor parallism
    fs_init.initialize_model_parallel(model_parallel_size_= world_size)
    torch.cuda.set_device(local_rank)

# time cycle
def get_time():
    torch.cuda.synchronize()
    return time.time()


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
        super(Attention_TP,  self).__init__()
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
    parser = argparse.ArgumentParser()
    parser.add_argument("-g", "--grad_checkpoint", action="store_true", help="Use gradient checkpointing")
    parser.add_argument("-x", "--mixed_precision", default="fp16", choices=["fp16", "bf16"], help="Mixed precision")
    parser.add_argument("-tp", "--tensor_parallelism", default=True, choices=[True, False], help="Tensor parallelism")
    parser.add_argument("-dp", "--data_parallelism", default=True, choices=[True, False], help="Data parallelism")
    args = parser.parse_args()
    
    # ==============================
    # 1. Tensor Parallism
    # ==============================
    device = int(os.environ['LOCAL_RANK'])
    world_size = int(os.environ['WORLD_SIZE'])
    
    batch_size, head, seq_length, dim = 64, 8, 128, 4096
    # atten no TP
    attn = Attention(batch_size, head, seq_length, dim)
    dk = dim // head # 512 / 8 = 64
    x = torch.randn(batch_size, head, seq_length, dim)
    # q, xk, v = torch.randn(batch_size, head, seq_length, dk), torch.randn(batch_size, head, seq_length, dk), torch.randn(batch_size, head, seq_length, dk)
    AttenStart = get_time()
    score = attn.forward(x)
    AttenStop = get_time()
    
    # atten with TP (tensor_parallelism)
    init_dist()
    x_tp = x.cuda(device=device)
    if args.tensor_parallelism == True:
        attn_tp = Attention_TP(batch_size, head, seq_length, dim).to(device)
    else:
        attn_tp = Attention(batch_size, head, seq_length, dim).to(device)
    dk = dim // head # 512 / 8 = 64
    AttenCPStart = get_time()
    score = attn_tp.forward(x_tp)
    AttenCPStop = get_time()
    print(f" Atten (No TP) {AttenStop - AttenStart}; Atten (with TP) {AttenCPStop - AttenCPStart}; \n")

    # ==============================
    # 2. Gradient Checkpoints
    # ==============================
    # save model
    model_name = 'Multi-Head_Atten'
    checkpoint_path = f'./checkpoints/{model_name}.pt'
    torch.save({
        'model_state_dict': attn_tp.state_dict()
    }, checkpoint_path)
    # load model & cont train
    attn_checkpoint = Attention_TP(batch_size, head, seq_length, dim).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    attn_checkpoint.load_state_dict(checkpoint['model_state_dict'])
    AttenCPStart = get_time()
    # attn_checkpoint.train()
    attn_checkpoint.forward(x_tp)
    AttenCPStop = get_time()
    print(f" Atten (load checkpoint) {AttenCPStop - AttenCPStart}; \n")

    # ==============================
    # 3. Mix percision
    # ==============================
    default_dtype = torch.float16 if args.mixed_precision == "fp16" else torch.bfloat16
    torch.set_default_dtype(default_dtype)
    x_mp = x.type(default_dtype).cuda(device=device)
    attn_mp = Attention_TP(batch_size, head, seq_length, dim).to(device)
    AttenMPStart = get_time()
    score = attn_mp.forward(x_mp)
    AttenMPStop = get_time()
    print(f" Atten (Mix percision) {AttenMPStop - AttenMPStart}; \n")   
    
    # ==============================
    # 4. Data Parallism
    # ==============================
    # 有4个GPU [0, 1, 2, 3],想实现DP + TP
    # Data Para: x = x1 + x2;  [0， 1] hold same x1, [2, 3] hold same x2
    # Tensor para： x1 [0, 1] , x2 [2, 3]
    # gradient accum; dx = （dx1 [0,1] + dx2 [2, 3]）// 2 
    # 两路DP
    
    torch.set_default_dtype(torch.float32)
    if args.data_parallelism == True:
        tensorList = torch.chunk(x, 2)
        x1, x2 = tensorList[0], tensorList[1]
        # print(f"len_tensorList:{len(tensorList)}, x1 shape{x1.shape}")
        attn_tp = Attention_TP(x1.shape[0], head, seq_length, dim).to(device)
        AttenDPStart = get_time()
        if device < world_size // 2:
            curr_x = x1.cuda(device=device)
            score = attn_tp.forward(curr_x)
        if device >= world_size // 2:
            curr_x = x2.cuda(device=device)
            score = attn_tp.forward(curr_x)
        AttenDPEnd = get_time()
        print(f" Atten (Data parallelism on {device}) {AttenDPEnd - AttenDPStart}; \n")   
    else:
        print(f" No Data parallelism; \n")
     
    # ==============================
    # 5. Test (test all func)
    # ==============================
    if args.data_parallelism == True and args.tensor_parallelism == True:
        # data init
        x = torch.randn(batch_size, head, seq_length, dim)  # 64, 8, 128, 4096
        y = torch.randn(batch_size, seq_length, dim)  # 64, 128, 4096
        x_tensorList = torch.chunk(x, 2)
        y_tensorList = torch.chunk(y, 2)
        x1, x2 = x_tensorList[0], x_tensorList[1] # 32, 8, 128, 4096
        y1, y2 = y_tensorList[0], y_tensorList[1] # 32, 128, 4096
        
        # TP class
        attn_tp = Attention_TP(x1.shape[0], head, seq_length, dim).to(device)
        loss_func = nn.CrossEntropyLoss().to(device)
        optimizer = optim.SGD([{"params":attn_tp.wq.parameters()}, {"params":attn_tp.wk.parameters()}, {"params":attn_tp.wv.parameters()}, {"params":attn_tp.wo.parameters()}], 
                              lr=0.01, momentum=0.9)
        
        # DP on 2 group
        if device < world_size // 2:
            curr_x = x1.cuda(device=device)
            curr_y = y1.cuda(device=device)
            score1 = attn_tp.forward(curr_x)
            loss = loss_func(score1, curr_y)
        else:
            # device >= world_size // 2:
            curr_x = x2.cuda(device=device)
            curr_y = y2.cuda(device=device)
            score2 = attn_tp.forward(curr_x)
            loss = loss_func(score2, curr_y)
            
        loss.retain_grad()
        loss.backward()
        print(f"Device {device}\n Gradient {loss.grad}\n")
        optimizer.step()
        optimizer.zero_grad()
        
        # Gradient Checkpoints
        if args.grad_checkpoint:
            # save model
            model_name = f'Multi-Head_Atten{device}'
            checkpoint_path = f'./checkpoints/{model_name}.pt'
            torch.save({
                'model_state_dict': attn_tp.state_dict()
            }, checkpoint_path)
            print(f"Save checkpoints{device} to {checkpoint_path}\n")
            # load model & cont train
            attn_checkpoint = Attention_TP(x1.shape[0], head, seq_length, dim).to(device)
            checkpoint = torch.load(checkpoint_path, map_location=device)
            print(f"Load checkpoints{device} from {checkpoint_path}\n")
            attn_checkpoint.load_state_dict(checkpoint['model_state_dict'])
            # attn_checkpoint.train()
            if device < world_size // 2:
                curr_x = x1.cuda(device=device)
                attn_checkpoint.forward(curr_x)
            else:
                curr_x = x2.cuda(device=device)
                attn_checkpoint.forward(curr_x)

        

        
    

    
    
    
       
    

if __name__ == "__main__":
    main()
    
    
    