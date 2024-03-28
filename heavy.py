import os
import tqdm
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim

from torch.nn.parallel import DistributedDataParallel as DDP


def setup(rank, world_size):
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()


class ToyModel(nn.Module):
    def __init__(self):
        super(ToyModel, self).__init__()
        self.net1 = nn.Linear(10, 10000)
        self.relu = nn.ReLU()
        self.net2 = nn.Linear(10000, 5)

    def forward(self, x):
        return self.net2(self.relu(self.net1(x)))


def demo_basic(rank, world_size):
    print(f"Running basic DDP example on rank {rank}.")
    print(f"Global rank {os.environ['RANK']}.")
    setup(int(os.environ['RANK']), world_size)

    # create model and move it to GPU with id rank
    model = ToyModel().to(rank)
    ddp_model = DDP(model, device_ids=[rank])

    loss_fn = nn.MSELoss()
    optimizer = optim.SGD(ddp_model.parameters(), lr=0.001)

    print("before start")
    for i in tqdm.trange(1000000, disable=rank != 0):
        optimizer.zero_grad()
        outputs = ddp_model(torch.randn(256, 1024, 10))
        labels = torch.randn(256, 5).to(rank)
        loss_fn(outputs.mean(-2), labels).backward()
        optimizer.step()

    cleanup()


if __name__ == '__main__':
    demo_basic(int(os.environ['LOCAL_RANK']), int(os.environ['WORLD_SIZE']))
