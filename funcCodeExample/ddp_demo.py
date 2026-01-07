# ddp_demo.py
import os
import torch.distributed as dist

# 启动命令： torchrun --nproc_per_node=1 ddp_demo.py
def main():
    dist.init_process_group(backend="gloo", init_method="env://")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ["LOCAL_RANK"])
    print(f"Hello from rank {rank}/{world_size}, local_rank={local_rank}")
    dist.destroy_process_group()

if __name__ == "__main__":
    main()