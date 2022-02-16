import os

import numpy as np
import pytest
from tbparse import SummaryReader
from torch.utils.tensorboard import SummaryWriter


@pytest.fixture
def prepare(testdir):
    # Ref: https://pytorch.org/docs/stable/tensorboard.html
    log_dir = os.path.join(testdir.tmpdir, 'run')

    img_batch = np.zeros((1, 16, 3, 100, 100))
    for i in range(16):
        img_batch[0, i, 0] = np.arange(0, 10000).reshape(100, 100) / 10000 / 16 * i
        img_batch[0, i, 1] = (1 - np.arange(0, 10000).reshape(100, 100) / 10000) / 16 * i

    writer = SummaryWriter(log_dir)
    writer.add_video('my_video', img_batch, 0)
    writer.close()

def test_log_dir(prepare, testdir):
    log_dir = os.path.join(testdir.tmpdir, 'run')
    reader = SummaryReader(log_dir, pivot=True)
    df = reader.tensors
    print(df.columns)
    print(df)
    assert False
