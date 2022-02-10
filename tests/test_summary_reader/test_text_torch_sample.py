import os

import pytest
from tbparse import SummaryReader
from torch.utils.tensorboard import SummaryWriter


@pytest.fixture
def prepare(testdir):
    # Ref: https://pytorch.org/docs/stable/tensorboard.html
    # Ref: https://github.com/WarrenWeckesser/wavio
    log_dir = os.path.join(testdir.tmpdir, 'run')

    writer = SummaryWriter(log_dir)
    writer.add_text('lstm', 'This is an lstm', 0)
    writer.add_text('rnn', 'This is an rnn', 10)
    writer.close()

def test_log_dir(prepare, testdir):
    log_dir = os.path.join(testdir.tmpdir, 'run')
    reader = SummaryReader(log_dir, pivot=True)
    df = reader.tensors
    print(df.columns)
    print(df)
    assert False
