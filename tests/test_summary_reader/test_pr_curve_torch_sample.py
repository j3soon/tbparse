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

@pytest.mark.skip(reason="add_pr_curve is not supported yet")
def test_log_dir(prepare, testdir):
    log_dir = os.path.join(testdir.tmpdir, 'run')
    reader = SummaryReader(log_dir, pivot=True)
    df = reader.tensors
    assert df.columns.tolist() == ['step', 'pr_curve']
    assert df.loc[0, 'step'] == 0
    # 127 is defind in `pr_curve` function in `torch/utils/tensorboard/summary.py`
    assert df.loc[0, 'pr_curve'].shape == (0, 127)
    assert False
