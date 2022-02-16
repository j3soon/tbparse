import os

import numpy as np
import pytest
from tbparse import SummaryReader
from torch.utils.tensorboard import SummaryWriter
from torch.utils.tensorboard._utils import convert_to_HWC


@pytest.fixture
def prepare(testdir):
    # Ref: https://pytorch.org/docs/stable/tensorboard.html
    log_dir = os.path.join(testdir.tmpdir, 'run')

    img_batch = np.zeros((16, 3, 100, 100))
    for i in range(16):
        img_batch[i, 0] = np.arange(0, 10000).reshape(100, 100) / 10000 / 16 * i
        img_batch[i, 1] = (1 - np.arange(0, 10000).reshape(100, 100) / 10000) / 16 * i

    writer = SummaryWriter(log_dir)
    writer.add_images('my_image_batch', img_batch, 0)
    writer.close()

def test_log_dir(prepare, testdir):
    log_dir = os.path.join(testdir.tmpdir, 'run')

    img_batch = np.zeros((16, 3, 100, 100))
    for i in range(16):
        img_batch[i, 0] = np.arange(0, 10000).reshape(100, 100) / 10000 / 16 * i
        img_batch[i, 1] = (1 - np.arange(0, 10000).reshape(100, 100) / 10000) / 16 * i
    img_batch_grid = convert_to_HWC(img_batch, 'NCHW')
    img_batch_grid_T_uint8 = (img_batch_grid.astype(np.float32) * 255).astype(np.uint8)

    reader = SummaryReader(log_dir, pivot=True)
    df = reader.images
    assert df.columns.to_list() == ['step', 'my_image_batch']
    assert df['step'].to_list() == [0]
    assert df['my_image_batch'][0].tolist() == img_batch_grid_T_uint8.tolist()
