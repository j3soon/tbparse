import os

import pytest
from tbparse import SummaryReader
from torch.utils.tensorboard import SummaryWriter


@pytest.fixture
def prepare(testdir):
    # Ref: https://pytorch.org/docs/stable/tensorboard.html
    log_dir = os.path.join(testdir.tmpdir, 'run')
    with SummaryWriter(os.path.join(log_dir, f'run0')) as w:
        for i in range(5):
            w.add_hparams({'lr': 0.1*i, 'bsize': i},
                          {'hparam/accuracy': 10*i, 'hparam/loss': 10*i})

def test_log_dir(prepare, testdir):
    log_dir = os.path.join(testdir.tmpdir, 'run')
    # default
    reader = SummaryReader(log_dir, pivot=True)
    df = reader.scalars
    assert df['hparam/accuracy'].to_list() == [[10*i for i in range(5)]]
    assert df['hparam/loss'].to_list() == [[10*i for i in range(5)]]
    hp = reader.hparams
    assert hp['lr'].to_list() == [[0.1*i for i in range(5)]]
    assert hp['bsize'].to_list() == [[i for i in range(5)]]
