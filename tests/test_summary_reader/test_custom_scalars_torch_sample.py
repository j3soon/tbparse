import os

import pytest
from tbparse import SummaryReader
from torch.utils.tensorboard import SummaryWriter


@pytest.fixture
def prepare(testdir):
    # Ref: https://pytorch.org/docs/stable/tensorboard.html
    log_dir = os.path.join(testdir.tmpdir, 'run')
    writer = SummaryWriter(log_dir)
    layout = {'Taiwan':{'twse':['Multiline',['twse/0050', 'twse/2330']]},
                'USA':{ 'dow':['Margin',   ['dow/aaa', 'dow/bbb', 'dow/ccc']],
                    'nasdaq':['Margin',   ['nasdaq/aaa', 'nasdaq/bbb', 'nasdaq/ccc']]}}

    writer.add_custom_scalars(layout)
    writer.close()

def test_log_dir(prepare, testdir):
    log_dir = os.path.join(testdir.tmpdir, 'run')
    reader = SummaryReader(log_dir, pivot=True)
    df = reader.tensors
    print(df.columns)
    print(df)
    assert False
