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
    labels = np.random.randint(2, size=100)  # binary label
    predictions = np.random.rand(100)
    writer = SummaryWriter(log_dir)
    writer.add_pr_curve('pr_curve', labels, predictions, 0)
    writer.close()

def test_log_dir(prepare, testdir):
    log_dir = os.path.join(testdir.tmpdir, 'run')
    reader = SummaryReader(log_dir, pivot=True)
    df = reader.tensors
    print(df.columns)
    print(df)
    print(df.loc[0, 'pr_curve'])
    assert df.columns.tolist() == ['step', 'pr_curve']
    assert df.loc[0, 'step'] == 0
    assert len(df.loc[0, 'pr_curve']) == 100
    assert False
