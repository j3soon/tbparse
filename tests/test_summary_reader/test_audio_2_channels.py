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

    x2 = np.sin(2*np.pi * (4*f) * t)
    assert x2.shape == (T*rate,)

    y = np.stack([x, x2])
    assert y.shape == (2, T*rate)
    return y, rate

def get_soundfile_compressed(tensor, sample_rate):
    # Defined in the `audio` function in `tensorboardX/summary.py`
    with io.BytesIO() as fio:
        soundfile.write(fio, tensor, samplerate=sample_rate, format='wav')
        audio_string = fio.getvalue()
    audio, rate = tf.audio.decode_wav(audio_string)
    value = audio.numpy()
    return value, rate

def get_encode_wav_compressed(tensor, sample_rate):
    # Defined in the `encode_wav` function in `tensorflow/python/ops/gen_audio_ops.py`
    audio_string = encode_wav(tensor, sample_rate)
    audio, rate = tf.audio.decode_wav(audio_string)
    value = audio.numpy()
    return value, rate

@pytest.fixture
def prepare(testdir):
    pass
    # Note: pytorch does not allow users to log audio with 2 channels.

def test_tensorboardX(prepare, testdir):
    # Note: tensorboardX automatically escapes special characters.
    # Prepare Log
    tmpdir_tbx = tempfile.TemporaryDirectory()
    log_dir_tbx = os.path.join(tmpdir_tbx.name, 'run')
    x, rate = get_audio()
    x = x.T
    x_compressed, rate_compressed = get_soundfile_compressed(x, rate)
    assert rate == rate_compressed
    writer = tensorboardX.SummaryWriter(log_dir_tbx)
    writer.add_audio('my_audio', x, 0, sample_rate=rate)
    writer.close()
    # (pivot) Parse & Compare
    df_tbx = SummaryReader(log_dir_tbx, pivot=True).audio
    assert df_tbx.columns.to_list() == ['step', 'my_audio']
    assert df_tbx['step'].to_list() == [0]
    assert df_tbx['my_audio'][0].tolist() == x_compressed.tolist()
    # (default) Parse & Compare
    df_tbx = SummaryReader(log_dir_tbx).audio
    assert df_tbx.columns.to_list() == ['step', 'tag', 'value']
    assert df_tbx['step'].to_list() == [0]
    assert df_tbx['tag'].to_list() == ['my_audio']
    assert df_tbx['value'][0].tolist() == x_compressed.tolist()

def test_tensorflow(prepare, testdir):
    # Prepare Log
    tmpdir_tf = tempfile.TemporaryDirectory()
    log_dir_tf = os.path.join(tmpdir_tf.name, 'run')
    x, rate = get_audio()
    x = x.T
    x_compressed, rate_compressed = get_encode_wav_compressed(x, rate)
    assert rate == rate_compressed
    x = np.expand_dims(x, axis=0).astype(np.float32)
    writer = tf.summary.create_file_writer(log_dir_tf)
    writer.set_as_default()
    tf.summary.audio('my_audio', x, rate, step=0)
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