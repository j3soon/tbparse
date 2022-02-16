import os

import numpy as np
import pytest
from tbparse import SummaryReader
from torch.utils.tensorboard import SummaryWriter


@pytest.fixture
def prepare(testdir):
    # Ref: https://numpy.org/neps/nep-0019-rng-policy.html#supporting-unit-tests
    np.random.seed(1234)
    # Ref: https://pytorch.org/docs/stable/tensorboard.html
    log_dir = os.path.join(testdir.tmpdir, 'run')
    writer = SummaryWriter(log_dir)
    for i in range(10):
        x = np.random.random(1000)
        writer.add_histogram('distribution centers', x + i, i)
    writer.close()

def test_log_dir(prepare, testdir):
    log_dir = os.path.join(testdir.tmpdir, 'run')
    reader = SummaryReader(log_dir, pivot=True)
    df = reader.histograms
    assert df.columns.tolist() == ['step', 'distribution centers/counts', 'distribution centers/limits']
    assert df['step'].tolist() == [i for i in range(10)]
    assert len(df['distribution centers/counts']) == 10
    assert len(df['distribution centers/limits']) == 10
