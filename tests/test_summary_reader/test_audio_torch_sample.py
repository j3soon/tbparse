import os

import numpy as np
import pytest
from tbparse import SummaryReader
from torch.utils.tensorboard import SummaryWriter


@pytest.fixture
def prepare(testdir):
    # Ref: https://pytorch.org/docs/stable/tensorboard.html
    # Ref: https://github.com/WarrenWeckesser/wavio
    log_dir = os.path.join(testdir.tmpdir, 'run')

    rate = 22050  # samples per second
    T = 3         # sample duration (seconds)
    f = 440.0     # sound frequency (Hz)
    t = np.linspace(0, T, T*rate, endpoint=False)
    x = np.sin(2*np.pi * f * t)

    x = np.expand_dims(x, axis=0)
    assert x.shape == (1, T*rate)

    writer = SummaryWriter(log_dir)
    writer.add_audio('my_audio', x, 0)
    writer.close()

def test_log_dir(prepare, testdir):
    log_dir = os.path.join(testdir.tmpdir, 'run')
    reader = SummaryReader(log_dir, pivot=True)
    df = reader.tensors
    print(df.columns)
    print(df)
    assert False
