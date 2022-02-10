import os

import matplotlib.pyplot as plt
import pytest
from tbparse import SummaryReader
from torch.utils.tensorboard import SummaryWriter


@pytest.fixture
def prepare(testdir):
    # Ref: https://pytorch.org/docs/stable/tensorboard.html
    # Ref: https://matplotlib.org/stable/tutorials/introductory/usage.html
    log_dir = os.path.join(testdir.tmpdir, 'run')

    fig, ax = plt.subplots()  # Create a figure containing a single axes.
    ax.plot([1, 2, 3, 4], [1, 4, 2, 3]);  # Plot some data on the axes.

    writer = SummaryWriter(log_dir)
    writer.add_figure('my_figure', fig)
    writer.close()

def test_log_dir(prepare, testdir):
    log_dir = os.path.join(testdir.tmpdir, 'run')
    reader = SummaryReader(log_dir, pivot=True)
    df = reader.tensors
    print(df.columns)
    print(df)
    assert False
