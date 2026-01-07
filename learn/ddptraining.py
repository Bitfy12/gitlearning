# train_ddp_env.py
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler, TensorDataset

#启动命令： torchrun --nproc_per_node=1 train_ddp_env.py
# nproc_per_node 表示每个节点的进程数，一般等于GPU数量
def setup():
    # 用 env:// 从 torchrun 传来的环境变量里自动读 rank/world_size
    # nccl 适用于 linux 系统NVIDIA GPU, windows 下可以用 gloo
    dist.init_process_group(backend="gloo", init_method="env://") 

def cleanup():
    dist.destroy_process_group()

def train():
    setup()
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ["LOCAL_RANK"])  # torchrun 会自动注入

    # 设置 GPU
    torch.cuda.set_device(local_rank)

    # 构造一个简单的模型
    model = nn.Linear(10, 1).cuda(local_rank)
    ddp_model = DDP(model, device_ids=[local_rank])

    # 数据集
    x = torch.randn(1000, 10)
    y = torch.randn(1000, 1)
    dataset = TensorDataset(x, y)

    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)
    dataloader = DataLoader(dataset, batch_size=32, sampler=sampler)

    optimizer = optim.SGD(ddp_model.parameters(), lr=0.01)

    for epoch in range(2):
        sampler.set_epoch(epoch)
        for batch_idx, (data, target) in enumerate(dataloader):
            data = data.cuda(local_rank)
            target = target.cuda(local_rank)

            optimizer.zero_grad()
            output = ddp_model(data)
            loss = nn.MSELoss()(output, target)
            loss.backward()
            optimizer.step()

            if batch_idx % 10 == 0 and rank == 0:  # 只在 rank=0 打印日志
                print(f"Epoch {epoch} | Batch {batch_idx} | Loss {loss.item():.4f}")

    cleanup()

if __name__ == "__main__":
    train()
