import os
from torch.utils.tensorboard import SummaryWriter

N_EVENTS = 10 ** 7
parent_dir = os.path.dirname(os.path.realpath(__file__))
log_dir = f"{parent_dir}/../benchmarks"
writer = SummaryWriter(os.path.join(log_dir, f'run_03'))
for i in range(N_EVENTS):
    writer.add_scalar('y=2x', i, i)
writer.close()
