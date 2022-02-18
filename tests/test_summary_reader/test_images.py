import os
import tempfile

import numpy as np
import pytest
import tensorboardX
import tensorflow as tf
from tbparse import SummaryReader
from torch.utils.tensorboard import SummaryWriter
from torch.utils.tensorboard._utils import convert_to_HWC


def get_images():
    img_batch = np.zeros((16, 3, 100, 100))
    for i in range(16):
        img_batch[i, 0] = np.arange(0, 10000).reshape(100, 100) / 10000 / 16 * i
        img_batch[i, 1] = (1 - np.arange(0, 10000).reshape(100, 100) / 10000) / 16 * i
    
    return img_batch

@pytest.fixture
def prepare(testdir):
    # Ref: https://pytorch.org/docs/stable/tensorboard.html
    log_dir = os.path.join(testdir.tmpdir, 'run')
    img_batch = get_images()
    writer = SummaryWriter(log_dir)
    writer.add_images('my_image_batch', img_batch, 0)
    writer.close()

def test_tensorboardX(prepare, testdir):
    # Note: tensorboardX automatically escapes special characters.
    # Prepare Log
    tmpdir_tbx = tempfile.TemporaryDirectory()
    log_dir_tbx = os.path.join(tmpdir_tbx.name, 'run')
    img_batch = get_images()
    writer = tensorboardX.SummaryWriter(log_dir_tbx)
    writer.add_images('my_image_batch', img_batch, 0)
    writer.close()
    # Note that this is different from torch since it does not cast with `.astype(np.float32)`
    from tensorboardX.utils import convert_to_HWC
    # The `convert_to_HWC` is requried due to the `image` function in `tensorboardX/summary.py`
    img_batch_grid = convert_to_HWC(img_batch, 'NCHW')
    img_batch_grid_uint8 = (img_batch_grid * 255).astype(np.uint8)
    # (pivot) Parse & Compare
    df_tbx = SummaryReader(log_dir_tbx, pivot=True).images
    assert df_tbx.columns.to_list() == ['step', 'my_image_batch']
    assert df_tbx['step'].to_list() == [0]
    assert df_tbx['my_image_batch'][0].shape == img_batch_grid_uint8.shape
    assert df_tbx['my_image_batch'][0].tolist() == img_batch_grid_uint8.tolist()
    # (default) Parse & Compare
    df_tbx = SummaryReader(log_dir_tbx).images
    assert df_tbx.columns.to_list() == ['step', 'tag', 'value']
    assert df_tbx['step'].to_list() == [0]
    assert df_tbx['tag'].to_list() == ['my_image_batch']
    assert df_tbx['value'][0].shape == img_batch_grid_uint8.shape
    assert df_tbx['value'][0].tolist() == img_batch_grid_uint8.tolist()

def test_tensorflow(prepare, testdir):
    # Prepare Log
    tmpdir_tf = tempfile.TemporaryDirectory()
    log_dir_tf = os.path.join(tmpdir_tf.name, 'run')
    img_batch = get_images()
    writer = tf.summary.create_file_writer(log_dir_tf)
    writer.set_as_default()
    tf.summary.image('my_image_batch', img_batch.transpose(0, 2, 3, 1), 0, max_outputs=img_batch.shape[0])
    writer.close()
    # Note that this is different from torch since it multiply with scale
    # `dtype.max + 0.5` in function `convert_image_dtype` in
    # `tensorflow/python/ops/image_ops_impl.py`
    img_batch_uint8 = (img_batch.transpose((0, 2, 3, 1)) * 255.5).astype(np.uint8)
    # (pivot) Parse & Compare
    df_tf = SummaryReader(log_dir_tf, pivot=True).tensors
    image_dict_arr = df_tf['my_image_batch'].apply(SummaryReader.tensor_to_image)
    df_tf['my_image_batch'] = image_dict_arr.apply(lambda x: x['image'])
    df_tf['my_image_batch/height'] = image_dict_arr.apply(lambda x: x['height'])
    df_tf['my_image_batch/width'] = image_dict_arr.apply(lambda x: x['width'])
    assert df_tf.columns.to_list() == ['step', 'my_image_batch', 'my_image_batch/height', 'my_image_batch/width']
    assert df_tf['step'].to_list() == [0]
    assert df_tf['my_image_batch'][0].shape == img_batch_uint8.shape
    assert df_tf['my_image_batch'][0].tolist() == img_batch_uint8.tolist()
    assert df_tf['my_image_batch/width'].to_list() == [100]
    assert df_tf['my_image_batch/height'].to_list() == [100]
    # (default) Parse & Compare
    df_tf = SummaryReader(log_dir_tf).tensors
    image_dict_arr = df_tf['value'].apply(SummaryReader.tensor_to_image)
    df_tf['value'] = image_dict_arr.apply(lambda x: x['image'])
    df_tf['height'] = image_dict_arr.apply(lambda x: x['height'])
    df_tf['width'] = image_dict_arr.apply(lambda x: x['width'])
    assert df_tf.columns.to_list() == ['step', 'tag', 'value', 'height', 'width']
    assert df_tf['step'].to_list() == [0]
    assert df_tf['tag'].to_list() == ['my_image_batch']
    assert df_tf['value'][0].shape == img_batch_uint8.shape
    assert df_tf['value'][0].tolist() == img_batch_uint8.tolist()
    assert df_tf['width'].to_list() == [100]
    assert df_tf['height'].to_list() == [100]

def test_log_dir(prepare, testdir):
    log_dir = os.path.join(testdir.tmpdir, 'run')
    img_batch = get_images()
    # The `convert_to_HWC` and `.astype(np.float32)` is requried due to the `image` function in
    # `torch/utils/tensorboard/summary.py
    img_batch_grid = convert_to_HWC(img_batch, 'NCHW')
    img_batch_grid_uint8 = (img_batch_grid.astype(np.float32) * 255).astype(np.uint8)

    reader = SummaryReader(log_dir, pivot=True)
    df = reader.images
    assert df.columns.to_list() == ['step', 'my_image_batch']
    assert df['step'].to_list() == [0]
    assert df['my_image_batch'][0].shape == img_batch_grid_uint8.shape
    assert df['my_image_batch'][0].tolist() == img_batch_grid_uint8.tolist()
