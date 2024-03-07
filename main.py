import math
import torch
import sys
import os
import time
import argparse
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
import warnings
warnings.filterwarnings('ignore')

from torch import nn
import torch.optim as optim
import torch.distributed as dist
import initialize as fs_init
from model import Attention, AttentionTP, AttentionTpCkpt



# dist env
def init_dist(args):
    rank = int(os.environ['RANK'])  # 当前进程的全局排名（Global Rank）
    local_rank = int(os.environ['LOCAL_RANK'])  # 表示当前进程在本地节点中的排名（Local Rank）。
    # single node GPU num :LOCAL RANK node0:{LOCAL_RANK0-3}, node1:{LOCAL_RANK 4-7}
    world_size = int(os.environ['WORLD_SIZE'])  # 表示分布式训练中总共的进程数。

    dist.init_process_group(world_size=world_size, rank=rank,
                            init_method="env://", backend="nccl")
    # model_parallel_size_= world_size，GPU全部用于tensor parallism
    fs_init.initialize_model_parallel(model_parallel_size_= args.tensor_parallel_size)
    torch.cuda.set_device(local_rank)

# time cycle
def get_time():
    torch.cuda.synchronize()
    return time.time()



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-g", "--grad_checkpoint", action="store_true", default=True, help="Use gradient checkpointing")
    parser.add_argument("-x", "--mixed_precision", default="fp16", choices=["fp16", "bf16"], help="Mixed precision")
    parser.add_argument("-tp", "--tensor_parallelism", action="store_true", default=True, help="Tensor parallelism")
    parser.add_argument("-dp", "--data_parallelism", action="store_true", default=True, help="Data parallelism")
    parser.add_argument("-tp_size", "--tensor_parallel_size", type=int, default=2, help="Tensor parallel Size")
    args = parser.parse_args()
    
    init_dist(args)  
    # ==============================
    # 1. Tensor Parallism
    # ==============================
    device = int(os.environ['LOCAL_RANK'])
    world_size = int(os.environ['WORLD_SIZE'])
    
    batch_size, head, seq_length, dim = 64, 8, 128, 4096
    # atten no TP
    attn = Attention(batch_size, head, seq_length, dim).to(device)
    dk = dim // head # 512 / 8 = 64
    x = torch.randn(batch_size, head, seq_length, dim)
    x = x.cuda(device=device)
    
    AttenStart = get_time()
    score = attn.forward(x)
    AttenStop = get_time()
    
    # atten with TP (tensor_parallelism)
    x_tp = x.cuda(device=device)
    if args.tensor_parallelism == True:
        attn_tp = AttentionTP(batch_size, head, seq_length, dim).to(device)
    else:
        attn_tp = Attention(batch_size, head, seq_length, dim).to(device)
    dk = dim // head # 512 / 8 = 64
    AttenCPStart = get_time()
    score = attn_tp.forward(x_tp)
    AttenCPStop = get_time()
    print(f"Atten (Base) on device {device} {AttenStop - AttenStart}; \n")
    print(f"Atten (with TP) {AttenCPStop - AttenCPStart}; \n")

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
    attn_checkpoint = AttentionTP(batch_size, head, seq_length, dim).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    attn_checkpoint.load_state_dict(checkpoint['model_state_dict'])
    AttenCPStart = get_time()
    # attn_checkpoint.train()
    attn_checkpoint.forward(x_tp)
    AttenCPStop = get_time()
    print(f"Atten (load checkpoint) {AttenCPStop - AttenCPStart}; \n")

    # ==============================
    # 3. Pure fp16 percision
    # ==============================
    default_dtype = torch.float16 if args.mixed_precision == "fp16" else torch.bfloat16
    torch.set_default_dtype(default_dtype)
    x_mp = x.type(default_dtype).cuda(device=device)
    attn_mp = AttentionTP(batch_size, head, seq_length, dim).to(device)
    AttenMPStart = get_time()
    score = attn_mp.forward(x_mp)
    AttenMPStop = get_time()
    print(f"Atten (Pure fp16) {AttenMPStop - AttenMPStart}; \n")   
    
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
        attn_tp = AttentionTP(x1.shape[0], head, seq_length, dim).to(device)
        AttenDPStart = get_time()
        if device < world_size // 2:
            curr_x = x1.cuda(device=device)
            score = attn_tp.forward(curr_x)
        if device >= world_size // 2:
            curr_x = x2.cuda(device=device)
            score = attn_tp.forward(curr_x)
        AttenDPEnd = get_time()
        print(f"Atten (Data parallelism on {device}) {AttenDPEnd - AttenDPStart}; \n")   
    else:
        print(f"No Data parallelism; \n")
    
    
    # ==============================
    # 5. Grad Checkpoint
    # ==============================
    x_tp = x.cuda(device=device)
    attn_tp_ckpt = AttentionTpCkpt(batch_size, head, seq_length, dim).to(device)
    AttenCPStart = get_time()
    score = attn_tp_ckpt.forward(x_tp)
    AttenCPStop = get_time()
    print(f"Atten (with TP and Checkpoint) {AttenCPStop - AttenCPStart}; \n")
    
    
    # ==============================
    # 6.Mix percision
    # ==============================
    x = torch.randn(batch_size, head, seq_length, dim).cuda(device=device)  # 64, 8, 128, 4096
    y = torch.randn(batch_size, seq_length, dim).cuda(device=device)  # 64, 128, 4096
    attn_tp = AttentionTP(batch_size, head, seq_length, dim).to(device)
    loss_func = nn.CrossEntropyLoss().to(device)
    optimizer = optim.SGD([{"params":attn_tp.wq.parameters()}, {"params":attn_tp.wk.parameters()}, {"params":attn_tp.wv.parameters()}, {"params":attn_tp.wo.parameters()}], 
                              lr=0.01, momentum=0.9)
    optimizer.zero_grad()
    with torch.autocast(device_type="cuda"):
        AttenMPStart = get_time()
        score = attn_tp.forward(x)
        AttenMPStop = get_time()
        print(f"Atten (AutoMP) {AttenMPStop - AttenMPStart}; \n") 
    
    
    # ==============================
    # 7. Test (test all func)
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
        attn_tp = AttentionTP(x1.shape[0], head, seq_length, dim).to(device)
        loss_func = nn.CrossEntropyLoss().to(device)
        optimizer = optim.SGD([{"params":attn_tp.wq.parameters()}, {"params":attn_tp.wk.parameters()}, {"params":attn_tp.wv.parameters()}, {"params":attn_tp.wo.parameters()}], 
                              lr=0.01, momentum=0.9)
        # attn_tp, optimizer = amp.initialize(attn_tp, optimizer, opt_level="O1")
        
        
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
            
        # loss.retain_grad()
        loss.backward()
        # print(f"Device {device}\n Gradient {loss.grad}\n")
        optimizer.step()
        optimizer.zero_grad()
        
        # Model Checkpoints
        if args.grad_checkpoint:
            # save model
            model_name = f'MultiHeadAtten{device}'
            checkpoint_path = f'./checkpoints/{model_name}.pt'
            torch.save({
                'model_state_dict': attn_tp.state_dict()
            }, checkpoint_path)
            print(f"Save checkpoints{device} to {checkpoint_path}\n")
            # load model & cont train
            attn_checkpoint = AttentionTP(x1.shape[0], head, seq_length, dim).to(device)
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
    
    
    