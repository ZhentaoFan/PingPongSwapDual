# 1.39 sec per iteration
# python -m torch.distributed.launch --nproc_per_node=4 BaselineTrain.py 

import os
import gc
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:64"
import time
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader

###############################################
# 定义 Dummy Dataset 和 32*6 层的 MLP 模型
###############################################

class DummyDataset(Dataset):
    def __init__(self, num_samples, input_dim, seq_len, device):
        self.num_samples = num_samples
        self.input_dim = input_dim
        self.seq_len = seq_len
        self.x = torch.randn(self.seq_len, self.input_dim).to(device)


    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        return self.x

class RMSNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-8):
        super(RMSNorm, self).__init__()
        self.eps = eps
        self.normalized_shape = normalized_shape

    def forward(self, x):
        norm = x.pow(2).mean(dim=-1, keepdim=True).sqrt()
        return x / (norm + self.eps)

class ExpandContractLayer(nn.Module):
    def __init__(self, hidden, expand_factor=4):
        super().__init__()
        self.expand_factor = expand_factor
        self.lin = nn.Linear(hidden, hidden)
        self.sft = nn.Softmax(dim=-1)
        self.silu = nn.SiLU()
        self.rms = RMSNorm(hidden)

    def forward(self, x):
        # x shape: (batch, seq_len, hidden_dim)
        x = self.silu(x)
        x = self.sft(x)
        for _ in range(20):
            x = self.lin(x)
        x = self.rms(x)
        return x

class SimpleMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=32):
        super(SimpleMLP, self).__init__()
        factor = 4
        layers = []
        # 第一层
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(ExpandContractLayer(hidden_dim))

        layers.append(nn.Linear(hidden_dim, hidden_dim))
        layers.append(ExpandContractLayer(hidden_dim))

        # 中间层（每个 block 包含两组 Linear+SiLU+CumSum+RMSNorm）
        for _ in range(num_layers - 2):
            layers.append(nn.Linear(hidden_dim, factor * hidden_dim))
            layers.append(ExpandContractLayer(factor * hidden_dim))

            layers.append(nn.Linear(factor * hidden_dim, hidden_dim))
            layers.append(ExpandContractLayer(hidden_dim))
    
        # 最后一层
        layers.append(nn.Linear(hidden_dim, hidden_dim))
        layers.append(ExpandContractLayer(hidden_dim))
        
        layers.append(nn.Linear(hidden_dim, output_dim))
        layers.append(ExpandContractLayer(hidden_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

###############################################
# 模型切分：按照两阶段分割，每个进程构造两个 local_model
###############################################
def split_model_two_phases(full_model, rank):
    layers = list(full_model.network.children())
    print('layers,', len(layers))
    # 假设总层数为32*8，将前16*8层作为 phase1，后16*8层作为 phase2
    first_half = layers[:len(layers)//2]
    second_half = layers[len(layers)//2:]
    # rank = int(os.environ["RANK"])
    start = rank * len(layers) // 8
    print('start,', start, rank)
    local_model1 = nn.Sequential(*first_half[start: start+ len(layers) // 8])
    local_model2 = nn.Sequential(*second_half[start: start+ len(layers) // 8])
    return local_model1, local_model2


def split_model_into_segments(full_model):
    layers = list(full_model.network.children())
    total_layers = len(layers)
    seg_size = total_layers // 8
    segments = []
    for i in range(8):
        # print("i * seg_size: (i + 1) * seg_size", i * seg_size, (i + 1) * seg_size)
        segment = nn.Sequential(*layers[i * seg_size: (i + 1) * seg_size])
        segments.append(segment)
    return segments


###############################################
# 定义发送/接收工具函数（保持不变）
###############################################
def send_tensor(tensor, dst, device, tag):
    dist.send(tensor, dst=dst, tag=tag)

def recv_tensor(shape, src, device, tag):
    tensor = torch.empty(shape, device=device)
    dist.recv(tensor, src=src, tag=tag)
    return tensor

###############################################
# 两阶段流水线（不是1F1B调度）实现
###############################################
def run_pipeline(rank, world_size, args):
    dist.init_process_group(backend="nccl", world_size=world_size, rank=rank)
    torch.manual_seed(0)
    device = torch.device(f"cuda:{rank}")

    # 构造完整模型，并切分为两个阶段的子模型
    full_model = SimpleMLP(args.input_dim, args.hidden_dim, args.output_dim, num_layers=args.num_layers)

    segments = split_model_into_segments(full_model)

    local_model1, local_model2 = segments[rank], segments[rank + 4]

    local_model1.to(device).train()
    local_model2.to(device).train()
    # optimizer1 = optim.Adam(local_model1.parameters(), lr=1e-3)
    # optimizer2 = optim.Adam(local_model2.parameters(), lr=1e-3)

    # 计算 phase1 的激活张量形状
    if rank == 0:
        dummy_in = torch.randn(args.batch_size, args.seq_len, args.input_dim, device=device)
        with torch.no_grad():
            dummy_out = local_model1(dummy_in)
        phase1_shape = dummy_out.shape
    else:
        for m in local_model1:
            if isinstance(m, nn.Linear):
                dummy_in = torch.randn(args.batch_size, args.seq_len, m.in_features, device=device)
                break
        phase1_shape = dummy_in.shape
    # 计算 phase2 的激活张量形状（各阶段第一层的 in_features）
    for m in local_model2:
        if isinstance(m, nn.Linear):
            dummy_in2 = torch.randn(args.batch_size, args.seq_len, m.in_features, device=device)
            break
    phase2_shape = dummy_in2.shape

    print("phase1_shape", phase1_shape)
    print("phase2_shape", phase2_shape)
    # 数据：仅 rank0 负责输入，rank3 负责计算 loss（target 数据）
    if rank == 0:
        dataset = DummyDataset(args.num_samples, args.input_dim, args.seq_len, device)
        dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
        data_iter = iter(dataloader)
    if rank == 3:
        target_dataset = DummyDataset(args.num_samples, args.input_dim, args.seq_len, device)
        target_dataloader = DataLoader(target_dataset, batch_size=args.batch_size, shuffle=True)
        target_iter = iter(target_dataloader)

    # 存储激活以便后向传播

    # print(f"[Rank {rank}] local_model1 = ")
    # for i, m in enumerate(local_model1):
    #     print(" ", i, repr(m))
    # print(f"[Rank {rank}] local_model2 = ")
    # for i, m in enumerate(local_model2):
    #     print(" ", i, repr(m))



    for iteration in range(args.num_iters):
        phase1_acts = []
        phase1_outs = []
        phase2_acts = []
        phase2_outs = []
        num_microbatches = args.num_microbatches

        if rank == 0 and iteration == 0:
            start_time = time.time()  # 开始计时

        for micro in range(num_microbatches):
            base_tag = iteration * 1000 + micro * 100
            #### Phase1 Forward: 顺序 rank0 -> rank1 -> rank2 -> rank3 ####
            if rank == 0:
                try:
                    x = next(data_iter)
                except StopIteration:
                    data_iter = iter(dataloader)
                    x = next(data_iter)
                x = x.to(device)
                act1 = local_model1(x)
                send_tensor(act1, dst=1, device=device, tag=base_tag + 0)
                phase1_outs.append(act1)
            elif rank in [1, 2]:
                act1 = recv_tensor(phase1_shape, src=rank-1, device=device, tag=base_tag + 0)
                act1.requires_grad_()
                phase1_acts.append(act1)
                act1_out = local_model1(act1)
                send_tensor(act1_out, dst=rank+1, device=device, tag=base_tag + 0)
                phase1_outs.append(act1_out)
            elif rank == 3:
                act1 = recv_tensor(phase1_shape, src=2, device=device, tag=base_tag + 0)
                act1.requires_grad_()
                phase1_acts.append(act1)
                act1_out = local_model1(act1)
                # Phase1 结束后，将输出发送给 phase2 开始设备（这里规定 phase2 的输入由 rank0 接收）
                send_tensor(act1_out, dst=0, device=device, tag=base_tag + 10)
                phase1_outs.append(act1_out)

        for micro in range(num_microbatches):
            base_tag = iteration * 1000 + micro * 100
            #### Phase2 Forward: 顺序 rank0 -> rank1 -> rank2 -> rank3 ####
            if rank == 0:
                act2_in = recv_tensor(phase2_shape, src=3, device=device, tag=base_tag + 10)
                # act2_in = act2_in.clone().detach().requires_grad_()
                act2_in.requires_grad_()
                phase2_acts.append(act2_in)
                act2_out = local_model2(act2_in)
                send_tensor(act2_out, dst=1, device=device, tag=base_tag + 10)
                phase2_outs.append(act2_out)
            elif rank in [1, 2]:
                act2_in = recv_tensor(phase2_shape, src=rank-1, device=device, tag=base_tag + 10)
                act2_in.requires_grad_()
                phase2_acts.append(act2_in)
                act2_out = local_model2(act2_in)
                send_tensor(act2_out, dst=rank+1, device=device, tag=base_tag + 10)
                phase2_outs.append(act2_out)
            elif rank == 3:
                act2_in = recv_tensor(phase2_shape, src=2, device=device, tag=base_tag + 10)
                act2_in.requires_grad_()
                phase2_acts.append(act2_in)
                act2_out = local_model2(act2_in)
                phase2_outs.append(act2_out)

        dist.barrier()
        if rank == 0:
            print(f"Iteration {iteration} forward finished.\n\n")
            end_time = time.time()
            print("Duration:", end_time - start_time)
            start_time = end_time

        for micro in range(num_microbatches):
            #### Phase2 Backward: 逆序 rank3 -> rank2 -> rank1 -> rank0 ####
            base_tag = iteration * 1000 + micro * 100
            if rank == 3:
                try:
                    y = next(target_iter)
                except StopIteration:
                    target_iter = iter(target_dataloader)
                    y = next(target_iter)
                y = y.to(device)
                act2_in = phase2_acts.pop(0)
                act2_out = phase2_outs.pop(0)
                loss = nn.MSELoss()(act2_out.view(-1, act2_out.size(-1)), y.view(-1, y.size(-1)))
                print(f"[Rank {rank}] Iteration {iteration} microbatch {micro}: loss = {loss.item():.4f}")

                loss.backward()
                # 将 phase2 backward 的起始梯度发送给 rank2
                # print('act2_in.grad', act2_in.grad)
                send_tensor(act2_in.grad, dst=2, device=device, tag=base_tag + 20)
            elif rank in [1, 2]:
                grad2 = recv_tensor(phase2_shape, src=rank+1, device=device, tag=base_tag + 20)
                act2_in = phase2_acts.pop(0)
                out2 = phase2_outs.pop(0)
                out2.backward(grad2)
                # print('act2_in.grad', act2_in.grad)
                send_tensor(act2_in.grad, dst=rank-1, device=device, tag=base_tag + 20)
            elif rank == 0:
                grad2 = recv_tensor(phase2_shape, src=1, device=device, tag=base_tag + 20)
                # print(grad2)
                # print("grad2.shape", grad2.shape)
                act2_in = phase2_acts.pop(0)
                # print("act2in.shape", act2_in.shape)
                out2 = phase2_outs.pop(0)
                out2.backward(grad2)
                # 将 phase2 backward 结束的梯度传给 phase1 backward 的起始设备（由 rank0 发送给 rank3）
                # print('act2in.grad', act2_in.grad)
                send_tensor(act2_in.grad, dst=3, device=device, tag=base_tag + 30)

                # dist.barrier()

        for micro in range(num_microbatches):
            base_tag = iteration * 1000 + micro * 100
            if rank == 3:
                grad1 = recv_tensor(phase1_shape, src=0, device=device, tag=base_tag + 30)
                act1 = phase1_acts.pop(0)
                out1 = phase1_outs.pop(0)
                out1.backward(grad1)
                send_tensor(act1.grad, dst=rank-1, device=device, tag=base_tag + 30)
            elif rank in [1, 2]:
                grad1 = recv_tensor(phase1_shape, src=rank+1, device=device, tag=base_tag + 30)
                act1 = phase1_acts.pop(0)
                out1 = phase1_outs.pop(0)
                out1.backward(grad1)
                send_tensor(act1.grad, dst=rank-1, device=device, tag=base_tag + 30)
            elif rank == 0:
                grad1 = recv_tensor(phase1_shape, src=1, device=device, tag=base_tag + 30)
                # print(grad1)
                out1 = phase1_outs.pop(0)
                out1.backward(grad1)
                # print(local_model1[0].weight.grad)

        # if iteration % args.grad_acc == args.grad_acc - 1:
        #     optimizer1.step()
        #     optimizer1.zero_grad()
        #     optimizer2.step()
        #     optimizer2.zero_grad()
            # gc.collect()
            # torch.cuda.empty_cache()

        dist.barrier()

        ##################################################
        ###  二周目 在这里让流水换个方向
        ##################################################

        reverse_phase1_acts = []
        reverse_phase1_outs = []
        reverse_phase2_acts = []
        reverse_phase2_outs = []

        for micro in range(num_microbatches):
            base_tag = iteration * 1000 + micro * 100 + 7
            #### Phase1 Forward: 顺序 rank0 -> rank1 -> rank2 -> rank3 ####
            if rank == 3:
                try:
                    x = next(target_iter)  # This is wrong and should be for temporary usage
                except StopIteration:
                    target_iter = iter(target_dataloader)
                    x = next(target_iter)
                x = x.to(device)
                act1 = local_model1(x)
                send_tensor(act1, dst=2, device=device, tag=base_tag + 0)
                reverse_phase1_outs.append(act1)
            elif rank in [1, 2]:
                act1 = recv_tensor(phase1_shape, src=rank+1, device=device, tag=base_tag + 0)
                act1.requires_grad_()
                reverse_phase1_acts.append(act1)
                act1_out = local_model1(act1)
                send_tensor(act1_out, dst=rank-1, device=device, tag=base_tag + 0)
                reverse_phase1_outs.append(act1_out)
            elif rank == 0:
                act1 = recv_tensor(phase1_shape, src=1, device=device, tag=base_tag + 0)
                act1.requires_grad_()
                reverse_phase1_acts.append(act1)
                act1_out = local_model1(act1)
                # Phase1 结束后，将输出发送给 phase2 开始设备（这里规定 phase2 的输入由 rank0 接收）
                send_tensor(act1_out, dst=3, device=device, tag=base_tag + 10)
                reverse_phase1_outs.append(act1_out)

        for micro in range(num_microbatches):
            base_tag = iteration * 1000 + micro * 100 + 7
            #### Phase2 Forward: 顺序 rank0 -> rank1 -> rank2 -> rank3 ####
            if rank == 3:
                act2_in = recv_tensor(phase2_shape, src=0, device=device, tag=base_tag + 10)
                # act2_in = act2_in.clone().detach().requires_grad_()
                act2_in.requires_grad_()
                reverse_phase2_acts.append(act2_in)
                act2_out = local_model2(act2_in)
                send_tensor(act2_out, dst=2, device=device, tag=base_tag + 10)
                reverse_phase2_outs.append(act2_out)
            elif rank in [1, 2]:
                act2_in = recv_tensor(phase2_shape, src=rank+1, device=device, tag=base_tag + 10)
                act2_in.requires_grad_()
                reverse_phase2_acts.append(act2_in)
                act2_out = local_model2(act2_in)
                send_tensor(act2_out, dst=rank-1, device=device, tag=base_tag + 10)
                reverse_phase2_outs.append(act2_out)
            elif rank == 0:
                act2_in = recv_tensor(phase2_shape, src=1, device=device, tag=base_tag + 10)
                act2_in.requires_grad_()
                reverse_phase2_acts.append(act2_in)
                act2_out = local_model2(act2_in)
                reverse_phase2_outs.append(act2_out)

        dist.barrier()

        for micro in range(num_microbatches):
            #### Phase2 Backward: 逆序 rank3 -> rank2 -> rank1 -> rank0 ####
            base_tag = iteration * 1000 + micro * 100 + 7
            if rank == 0:
                try:
                    y = next(data_iter)  # This is wrong and should be for temporary usage
                except StopIteration:
                    data_iter = iter(dataloader)
                    y = next(data_iter)
                y = y.to(device)
                act2_in = reverse_phase2_acts.pop(0)
                act2_out = reverse_phase2_outs.pop(0)
                loss = nn.MSELoss()(act2_out.view(-1, act2_out.size(-1)), y.view(-1, y.size(-1)))
                print(f"[Rank {rank}] Iteration {iteration} microbatch {micro}: loss = {loss.item():.4f}")

                loss.backward()
                # 将 phase2 backward 的起始梯度发送给 rank2
                # print('act2_in.grad', act2_in.grad)
                send_tensor(act2_in.grad, dst=1, device=device, tag=base_tag + 20)
            elif rank in [1, 2]:
                grad2 = recv_tensor(phase2_shape, src=rank-1, device=device, tag=base_tag + 20)
                act2_in = reverse_phase2_acts.pop(0)
                out2 = reverse_phase2_outs.pop(0)
                out2.backward(grad2)
                # print('act2_in.grad', act2_in.grad)
                send_tensor(act2_in.grad, dst=rank+1, device=device, tag=base_tag + 20)
            elif rank == 3:
                grad2 = recv_tensor(phase2_shape, src=2, device=device, tag=base_tag + 20)
                # print(grad2)
                # print("grad2.shape", grad2.shape)
                act2_in = reverse_phase2_acts.pop(0)
                # print("act2in.shape", act2_in.shape)
                out2 = reverse_phase2_outs.pop(0)
                out2.backward(grad2)
                # 将 phase2 backward 结束的梯度传给 phase1 backward 的起始设备（由 rank0 发送给 rank3）
                # print('act2in.grad', act2_in.grad)
                send_tensor(act2_in.grad, dst=0, device=device, tag=base_tag + 30)

                # dist.barrier()

        for micro in range(num_microbatches):
            base_tag = iteration * 1000 + micro * 100 + 7
            if rank == 0:
                grad1 = recv_tensor(phase1_shape, src=3, device=device, tag=base_tag + 30)
                act1 = reverse_phase1_acts.pop(0)
                out1 = reverse_phase1_outs.pop(0)
                out1.backward(grad1)
                send_tensor(act1.grad, dst=rank+1, device=device, tag=base_tag + 30)
            elif rank in [1, 2]:
                grad1 = recv_tensor(phase1_shape, src=rank-1, device=device, tag=base_tag + 30)
                act1 = reverse_phase1_acts.pop(0)
                out1 = reverse_phase1_outs.pop(0)
                out1.backward(grad1)
                send_tensor(act1.grad, dst=rank+1, device=device, tag=base_tag + 30)
            elif rank == 3:
                grad1 = recv_tensor(phase1_shape, src=2, device=device, tag=base_tag + 30)
                # print(grad1)
                out1 = reverse_phase1_outs.pop(0)
                out1.backward(grad1)

        dist.barrier()


        ##################################################
        ###  在这里让流水换个方向
        ##################################################


        # if rank == 0:
            # print(f"Iteration {iteration} completed.\n\n")
    if rank == 0:
        print("两阶段 PP 流水线训练全部完成。")

###############################################
# 主入口
###############################################
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--init_method", type=str, default="tcp://127.0.0.1:12345")
    parser.add_argument("--num_samples", type=int, default=1000000)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--seq_len", type=int, default=1024)
    parser.add_argument("--num_microbatches", type=int, default=4)
    parser.add_argument("--num_iters", type=int, default=30)
    parser.add_argument("--input_dim", type=int, default=256)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--output_dim", type=int, default=256)
    parser.add_argument("--num_layers", type=int, default=32)
    parser.add_argument("--grad_acc", type=int, default=6)
    parser.add_argument("--local_rank", type=int, default=0)
    args, unknown = parser.parse_known_args()

    world_size = 4
    rank = int(os.environ["RANK"])
    run_pipeline(rank, world_size, args)
