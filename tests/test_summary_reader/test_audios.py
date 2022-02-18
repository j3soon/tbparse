import io
import os
import tempfile
import wave

import numpy as np
import pytest
import soundfile
import tensorboardX
import tensorflow as tf
from tbparse import SummaryReader
from tensorboard.backend.event_processing.event_accumulator import AudioEvent
from tensorflow.python.ops.gen_audio_ops import encode_wav
from torch.utils.tensorboard import SummaryWriter


def get_audio():
    # Ref: https://github.com/WarrenWeckesser/wavio
    rate = 22050  # samples per second
    T = 3         # sample duration (seconds)
    f = 440.0     # sound frequency (Hz)
    t = np.linspace(0, T, T*rate, endpoint=False)

    x = np.sin(2*np.pi * f * t)
    assert x.shape == (T*rate,)
    x2 = np.sin(2*np.pi * (2*f) * t)
    assert x2.shape == (T*rate,)

    y = np.sin(2*np.pi * f * t)
    assert y.shape == (T*rate,)
    y2 = np.sin(2*np.pi * (4*f) * t)
    assert y2.shape == (T*rate,)

    z1 = np.stack([x, x2], axis=1)
    z2 = np.stack([y, y2], axis=1)
    z = np.stack([z1, z2], axis=0)
    assert z.shape == (2, T*rate, 2)
    return z, rate

def get_encode_wav_compressed(tensor, sample_rate):
    # Defined in the `encode_wav` function in `tensorflow/python/ops/gen_audio_ops.py`
    audio_string = encode_wav(tensor, sample_rate)
    audio, rate = tf.audio.decode_wav(audio_string)
    value = audio.numpy()
    return value, rate

@pytest.fixture
def prepare(testdir):
    pass
    # Note: pytorch does not allow users to log multiple audio.

def test_tensorboardX(prepare, testdir):
    pass
    # Note: tensorboardX doesn't support logging multiple audio.

def test_tensorflow(prepare, testdir):
    # Prepare Log
    tmpdir_tf = tempfile.TemporaryDirectory()
    log_dir_tf = os.path.join(tmpdir_tf.name, 'run')
    x, rate = get_audio()
    x_compressed = np.ndarray(x.shape)
    x_compressed[0], rate_compressed = get_encode_wav_compressed(x[0], rate)
    assert rate == rate_compressed
    x_compressed[1], rate_compressed = get_encode_wav_compressed(x[1], rate)
    assert rate == rate_compressed
    x = x.astype(np.float32)
    writer = tf.summary.create_file_writer(log_dir_tf)
    writer.set_as_default()
    tf.summary.audio('my_audio', x, rate, step=0, max_outputs=x.shape[0])
    writer.close()
    # (default) Parse & Compare
    df_tf = SummaryReader(log_dir_tf).tensors
    audio_dict_arr = df_tf['value'].apply(SummaryReader.tensor_to_audio)
    df_tf['value'] = audio_dict_arr.apply(lambda x: x['audio'])
    assert df_tf.columns.to_list() == ['step', 'tag', 'value']
    assert df_tf['step'].to_list() == [0]
    assert df_tf['tag'].to_list() == ['my_audio']
    assert df_tf['value'][0].tolist() == x_compressed.tolist()
    # (pivot) Parse & Compare
    df_tf = SummaryReader(log_dir_tf, pivot=True).tensors
    audio_dict_arr = df_tf['my_audio'].apply(SummaryReader.tensor_to_audio)
    df_tf['my_audio'] = audio_dict_arr.apply(lambda x: x['audio'])
    assert df_tf.columns.to_list() == ['step', 'my_audio']
    assert df_tf['step'].to_list() == [0]
    assert df_tf['my_audio'][0].tolist() == x_compressed.tolist()

def test_log_dir(prepare, testdir):
    pass
    # Note: pytorch does not allow users to log audio with 2 channels.
