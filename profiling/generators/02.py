import os
from torch.utils.tensorboard import SummaryWriter

N_RUNS = 10
N_EVENTS = 10 ** 5
parent_dir = os.path.dirname(os.path.realpath(__file__))
log_dir = f"{parent_dir}/../benchmarks"
for i in range(N_RUNS):
    writer = SummaryWriter(os.path.join(log_dir, f'run_02'))
    for j in range(N_EVENTS):
        writer.add_scalar('y=2x', j, j)
    writer.close()
