import os

import numpy as np
import pytest
from tbparse import SummaryReader
from torch.utils.tensorboard import SummaryWriter


@pytest.fixture
def prepare(testdir):
    # Ref: https://pytorch.org/docs/stable/tensorboard.html
    log_dir = os.path.join(testdir.tmpdir, 'run')
    writer = SummaryWriter(log_dir)
    x = range(100)
    for i in x:
        writer.add_scalar('y=2x', i * 2, i)
    writer.add_text('text', 'lorem ipsum', 0)
    writer.close()

def test_log_dir(prepare, testdir):
    log_dir = os.path.join(testdir.tmpdir, 'run')
    reader = SummaryReader(log_dir, pivot=True)
    df = reader.scalars
    assert df.columns.tolist() == ['step', 'y=2x']
    assert df['step'].to_list() == [i for i in range(100)]
    assert df['y=2x'].to_list() == [i*2 for i in range(100)]
    with pytest.raises(ModuleNotFoundError):
        df = reader.text
    with pytest.raises(ModuleNotFoundError):
        reader = SummaryReader(log_dir, pivot=True, event_types={'scalars', 'text'})
