import os
import tempfile

import numpy as np
import pytest
import tensorboardX
import tensorflow as tf
from tbparse import SummaryReader
from tensorboard.backend.event_processing.event_accumulator import ImageEvent
from torch.utils.tensorboard import SummaryWriter


def get_images():
    img = np.zeros((3, 100, 100))
    img[0] = np.arange(0, 10000).reshape(100, 100) / 10000
    img[1] = 1 - np.arange(0, 10000).reshape(100, 100) / 10000

    img_HWC = np.zeros((100, 100, 3))
    img_HWC[:, :, 0] = np.arange(0, 10000).reshape(100, 100) / 10000
    img_HWC[:, :, 1] = 1 - np.arange(0, 10000).reshape(100, 100) / 10000
    
    return img, img_HWC

@pytest.fixture
def prepare(testdir):
    # Ref: https://pytorch.org/docs/stable/tensorboard.html
    log_dir = os.path.join(testdir.tmpdir, 'run')

    img, img_HWC = get_images()
    writer = SummaryWriter(log_dir)
    writer.add_image('my_image', img, 0)
    # If you have non-default dimension setting, set the dataformats argument.
    writer.add_image('my_image_HWC', img_HWC, 0, dataformats='HWC')
    writer.close()

def test_tensorboardX(prepare, testdir):
    # Note: tensorboardX automatically escapes special characters.
    # Prepare Log
    tmpdir_tbx = tempfile.TemporaryDirectory()
    log_dir_tbx = os.path.join(tmpdir_tbx.name, 'run')
    img, img_HWC = get_images()
    writer = tensorboardX.SummaryWriter(log_dir_tbx)
    writer.add_image('my_image', img, 0)
    # If you have non-default dimension setting, set the dataformats argument.
    writer.add_image('my_image_HWC', img_HWC, 0, dataformats='HWC')
    writer.close()
    # Note that this is different from torch since it does not cast with `.astype(np.float32)`
    img_T_uint8 = (img.transpose((1, 2, 0)) * 255).astype(np.uint8)
    img_HWC_uint8 = (img_HWC * 255).astype(np.uint8)
    # (pivot) Parse & Compare
    df_tbx = SummaryReader(log_dir_tbx, pivot=True).images
    assert df_tbx.columns.to_list() == ['step', 'my_image', 'my_image_HWC']
    assert df_tbx['step'].to_list() == [0]
    assert df_tbx['my_image'][0].shape == img_T_uint8.shape
    assert df_tbx['my_image'][0].tolist() == img_T_uint8.tolist()
    assert df_tbx['my_image_HWC'][0].shape == img_HWC_uint8.shape
    assert df_tbx['my_image_HWC'][0].tolist() == img_HWC_uint8.tolist()
    # (default) Parse & Compare
    df_tbx = SummaryReader(log_dir_tbx).images
    assert df_tbx.columns.to_list() == ['step', 'tag', 'value']
    assert df_tbx['step'].to_list() == [0] * 2
    assert df_tbx['tag'].to_list() == ['my_image', 'my_image_HWC']
    assert df_tbx['value'][0].shape == img_T_uint8.shape
    assert df_tbx['value'][0].tolist() == img_T_uint8.tolist()
    assert df_tbx['value'][1].shape == img_HWC_uint8.shape
    assert df_tbx['value'][1].tolist() == img_HWC_uint8.tolist()

def test_tensorflow(prepare, testdir):
    # Prepare Log
    tmpdir_tf = tempfile.TemporaryDirectory()
    log_dir_tf = os.path.join(tmpdir_tf.name, 'run')
    img, img_HWC = get_images()
    writer = tf.summary.create_file_writer(log_dir_tf)
    writer.set_as_default()
    tf.summary.image('my_image', np.expand_dims(img.transpose(1, 2, 0), axis=0), 0)
    tf.summary.image('my_image_HWC', np.expand_dims(img_HWC, axis=0), 0)
    writer.close()
    # Note that this is different from torch since it multiply with scale
    # `dtype.max + 0.5` in function `convert_image_dtype` in
    # `tensorflow/python/ops/image_ops_impl.py`
    img_T_uint8 = (img.transpose((1, 2, 0)) * 255.5).astype(np.uint8)
    img_HWC_uint8 = (img_HWC * 255.5).astype(np.uint8)
    # (pivot) Parse & Compare
    df_tf = SummaryReader(log_dir_tf, pivot=True).tensors
    image_dict_arr = df_tf['my_image'].apply(SummaryReader.tensor_to_image)
    df_tf['my_image'] = image_dict_arr.apply(lambda x: x['image'])
    df_tf['my_image/height'] = image_dict_arr.apply(lambda x: x['height'])
    df_tf['my_image/width'] = image_dict_arr.apply(lambda x: x['width'])
    image_dict_arr = df_tf['my_image_HWC'].apply(SummaryReader.tensor_to_image)
    df_tf['my_image_HWC'] = image_dict_arr.apply(lambda x: x['image'])
    df_tf['my_image_HWC/height'] = image_dict_arr.apply(lambda x: x['height'])
    df_tf['my_image_HWC/width'] = image_dict_arr.apply(lambda x: x['width'])
    assert df_tf.columns.to_list() == ['step', 'my_image', 'my_image_HWC', 'my_image/height', 'my_image/width', 'my_image_HWC/height', 'my_image_HWC/width']
    assert df_tf['step'].to_list() == [0]
    assert df_tf['my_image'][0].shape == img_T_uint8.shape
    assert df_tf['my_image'][0].tolist() == img_T_uint8.tolist()
    assert df_tf['my_image_HWC'][0].shape == img_HWC_uint8.shape
    assert df_tf['my_image_HWC'][0].tolist() == img_HWC_uint8.tolist()
    assert df_tf['my_image/width'].to_list() == [100]
    assert df_tf['my_image/height'].to_list() == [100]
    assert df_tf['my_image_HWC/width'].to_list() == [100]
    assert df_tf['my_image_HWC/height'].to_list() == [100]
    # (default) Parse & Compare
    df_tf = SummaryReader(log_dir_tf).tensors
    image_dict_arr = df_tf['value'].apply(SummaryReader.tensor_to_image)
    df_tf['value'] = image_dict_arr.apply(lambda x: x['image'])
    df_tf['height'] = image_dict_arr.apply(lambda x: x['height'])
    df_tf['width'] = image_dict_arr.apply(lambda x: x['width'])
    assert df_tf.columns.to_list() == ['step', 'tag', 'value', 'height', 'width']
    assert df_tf['step'].to_list() == [0] * 2
    assert df_tf['tag'].to_list() == ['my_image', 'my_image_HWC']
    assert df_tf['value'][0].shape == img_T_uint8.shape
    assert df_tf['value'][0].tolist() == img_T_uint8.tolist()
    assert df_tf['value'][1].shape == img_HWC_uint8.shape
    assert df_tf['value'][1].tolist() == img_HWC_uint8.tolist()
    assert df_tf['width'].to_list() == [100] * 2
    assert df_tf['height'].to_list() == [100] * 2

def get_tmpdir_info(tmpdir):
    log_dir = os.path.join(tmpdir, 'run')
    dirs = os.listdir(log_dir)
    assert len(dirs) == 1
    event_filename = dirs[0]
    event_file = os.path.join(log_dir, event_filename)
    d = {
        'log_dir': log_dir,
        'event_file': event_file,
        'event_filename': event_filename,
    }
    return d

def test_event_file_raw(prepare, testdir):
    tmpinfo = get_tmpdir_info(testdir.tmpdir)
    img, img_HWC = get_images()
    img_T_uint8 = (img.astype(np.float32).transpose((1, 2, 0)) * 255).astype(np.uint8)
    img_HWC_uint8 = (img_HWC.astype(np.float32) * 255).astype(np.uint8)
    reader = SummaryReader(tmpinfo["event_file"], pivot=True)
    # Test raw functions
    # - Test `raw_tags` and `get_raw_tags`
    assert reader.raw_tags == reader.get_raw_tags()
    assert reader.raw_tags['images'] == reader.get_raw_tags('images')
    assert set(reader.raw_tags['images']) == {'my_image', 'my_image_HWC'}
    # - Test `raw_events` and `get_raw_events`
    assert reader.raw_events == reader.get_raw_events()
    assert reader.raw_events['images'] == reader.get_raw_events('images')
    assert reader.raw_events['images']['my_image'] == reader.get_raw_events('images', 'my_image')
    assert reader.raw_events['images']['my_image_HWC'] == reader.get_raw_events('images', 'my_image_HWC')
    # - (my_image) Test raw event count & type
    events = reader.get_raw_events('images', 'my_image')
    assert len(events) == 1
    e = events[0]
    assert type(e) == ImageEvent
    value = tf.image.decode_image(e.encoded_image_string).numpy()
    assert (e.step, value.tolist()) == (0, img_T_uint8.tolist())
    assert (e.width, e.height) == (100, 100)
    # - (my_image_HWC) Test raw event count & type
    events = reader.get_raw_events('images', 'my_image_HWC')
    assert len(events) == 1
    e = events[0]
    assert type(e) == ImageEvent
    value = tf.image.decode_image(e.encoded_image_string).numpy()
    assert (e.step, value.tolist()) == (0, img_HWC_uint8.tolist())
    assert (e.width, e.height) == (100, 100)

def check_others(reader):
    assert len(reader.scalars) == 0
    assert len(reader.tensors) == 0
    assert len(reader.histograms) == 0
    assert len(reader.hparams) == 0
    assert len(reader.text) == 0

def test_log_dir(prepare, testdir):
    tmpinfo = get_tmpdir_info(testdir.tmpdir)
    img, img_HWC = get_images()
    # The `.astype(np.float32)` is requried due to the `image` function in
    # `torch/utils/tensorboard/summary.py
    img_T_uint8 = (img.astype(np.float32).transpose((1, 2, 0)) * 255).astype(np.uint8)
    img_HWC_uint8 = (img_HWC.astype(np.float32) * 255).astype(np.uint8)
    # Test pivot
    reader = SummaryReader(tmpinfo["log_dir"], pivot=True, extra_columns={
                           'dir_name', 'file_name'})
    assert len(reader.children) == 1
    df = reader.images
    assert df.columns.to_list() == ['step', 'my_image', 'my_image_HWC', 'dir_name', 'file_name']
    assert df['step'].to_list() == [0]
    assert df['my_image'][0].shape == img_T_uint8.shape
    assert df['my_image'][0].tolist() == img_T_uint8.tolist()
    assert df['my_image_HWC'][0].shape == img_HWC_uint8.shape
    assert df['my_image_HWC'][0].tolist() == img_HWC_uint8.tolist()
    assert df['dir_name'].to_list() == ['']
    assert df['file_name'].to_list() == [tmpinfo["event_filename"]]
    check_others(reader)
    # Test default
    reader = SummaryReader(tmpinfo["log_dir"], extra_columns={
                           'dir_name', 'file_name'})
    df = reader.images
    assert df.columns.to_list() == ['step', 'tag', 'value', 'dir_name', 'file_name']
    assert df['step'].to_list() == [0] * 2
    assert df['tag'].to_list() == ['my_image', 'my_image_HWC']
    assert df['value'][0].shape == img_T_uint8.shape
    assert df['value'][0].tolist() == img_T_uint8.tolist()
    assert df['value'][1].shape == img_HWC_uint8.shape
    assert df['value'][1].tolist() == img_HWC_uint8.tolist()
    assert df['dir_name'].to_list() == [''] * 2
    assert df['file_name'].to_list() == [tmpinfo["event_filename"]] * 2
    check_others(reader)
    # Test pivot & all columns
    reader = SummaryReader(tmpinfo["log_dir"], pivot=True, extra_columns={
                           'width', 'height', 'dir_name', 'file_name'})
    df = reader.images
    assert len(reader.children) == 1
    assert df.columns.to_list() == ['step', 'my_image', 'my_image/height', 'my_image/width', 'my_image_HWC', 'my_image_HWC/height', 'my_image_HWC/width', 'dir_name', 'file_name']
    assert df['step'].to_list() == [0]
    assert df['my_image'][0].shape == img_T_uint8.shape
    assert df['my_image'][0].tolist() == img_T_uint8.tolist()
    assert df['my_image_HWC'][0].shape == img_HWC_uint8.shape
    assert df['my_image_HWC'][0].tolist() == img_HWC_uint8.tolist()
    assert df['my_image/width'].to_list() == [100]
    assert df['my_image/height'].to_list() == [100]
    assert df['my_image_HWC/width'].to_list() == [100]
    assert df['my_image_HWC/height'].to_list() == [100]
    assert df['dir_name'].to_list() == ['']
    assert df['file_name'].to_list() == [tmpinfo["event_filename"]]
    check_others(reader)
    # Test all columns
    reader = SummaryReader(tmpinfo["log_dir"], extra_columns={
                           'width', 'height', 'dir_name', 'file_name'})
    df = reader.images
    assert df.columns.to_list() == ['step', 'tag', 'value', 'height', 'width', 'dir_name', 'file_name']
    assert df['step'].to_list() == [0] * 2
    assert df['tag'].to_list() == ['my_image', 'my_image_HWC']
    assert df['value'][0].shape == img_T_uint8.shape
    assert df['value'][0].tolist() == img_T_uint8.tolist()
    assert df['value'][1].shape == img_HWC_uint8.shape
    assert df['value'][1].tolist() == img_HWC_uint8.tolist()
    assert df['width'].to_list() == [100] * 2
    assert df['height'].to_list() == [100] * 2
    assert df['dir_name'].to_list() == [''] * 2
    assert df['file_name'].to_list() == [tmpinfo["event_filename"]] * 2
    check_others(reader)