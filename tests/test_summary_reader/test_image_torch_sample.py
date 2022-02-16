import os

import numpy as np
import pytest
from tbparse import SummaryReader
from torch.utils.tensorboard import SummaryWriter


@pytest.fixture
def prepare(testdir):
    # Ref: https://pytorch.org/docs/stable/tensorboard.html
    log_dir = os.path.join(testdir.tmpdir, 'run')

    img = np.zeros((3, 100, 100))
    img[0] = np.arange(0, 10000).reshape(100, 100) / 10000
    img[1] = 1 - np.arange(0, 10000).reshape(100, 100) / 10000

    img_HWC = np.zeros((100, 100, 3))
    img_HWC[:, :, 0] = np.arange(0, 10000).reshape(100, 100) / 10000
    img_HWC[:, :, 1] = 1 - np.arange(0, 10000).reshape(100, 100) / 10000

    writer = SummaryWriter(log_dir)
    writer.add_image('my_image', img, 0)

    # If you have non-default dimension setting, set the dataformats argument.
    writer.add_image('my_image_HWC', img_HWC, 0, dataformats='HWC')
    writer.close()

def test_log_dir(prepare, testdir):
    log_dir = os.path.join(testdir.tmpdir, 'run')

    img = np.zeros((3, 100, 100))
    img[0] = np.arange(0, 10000).reshape(100, 100) / 10000
    img[1] = 1 - np.arange(0, 10000).reshape(100, 100) / 10000

    img_HWC = np.zeros((100, 100, 3))
    img_HWC[:, :, 0] = np.arange(0, 10000).reshape(100, 100) / 10000
    img_HWC[:, :, 1] = 1 - np.arange(0, 10000).reshape(100, 100) / 10000

    # The `.astype(np.float32)` is requried due to the `image` function in
    # `torch/utils/tensorboard/summary.py
    img_T_uint8 = (img.astype(np.float32).transpose((1, 2, 0)) * 255).astype(np.uint8)
    img_HWC_uint8 = (img_HWC.astype(np.float32) * 255).astype(np.uint8)

    reader = SummaryReader(log_dir, pivot=True)
    df = reader.images
    assert df.columns.to_list() == ['step', 'my_image', 'my_image_HWC']
    assert df['step'].to_list() == [0]
    assert df['my_image'][0].shape == img_T_uint8.shape
    assert df['my_image'][0].tolist() == img_T_uint8.tolist()
    assert df['my_image_HWC'][0].shape == img_HWC_uint8.shape
    assert df['my_image_HWC'][0].tolist() == img_HWC_uint8.tolist()