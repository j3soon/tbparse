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

    x = np.expand_dims(x, axis=0)
    assert x.shape == (1, T*rate)
    
    return x, rate

def get_wave_compressed(tensor, sample_rate):
    # Defined in the `audio` function in `torch/utils/tensorboard/summary.py`
    tensor = tensor.squeeze()
    tensor = (tensor * np.iinfo(np.int16).max).astype('<i2')
    fio = io.BytesIO()
    wave_write = wave.open(fio, 'wb')
    wave_write.setnchannels(1)
    wave_write.setsampwidth(2)
    wave_write.setframerate(sample_rate)
    wave_write.writeframes(tensor)
    wave_write.close()
    audio_string = fio.getvalue()
    fio.close()
    audio, rate = tf.audio.decode_wav(audio_string)
    value = audio.numpy()
    return value, rate

def get_soundfile_compressed(tensor, sample_rate):
    # Defined in the `encode_wav` function in `tensorflow/python/ops/gen_audio_ops.py`
    with io.BytesIO() as fio:
        soundfile.write(fio, tensor, samplerate=sample_rate, format='wav')
        audio_string = fio.getvalue()
    audio, rate = tf.audio.decode_wav(audio_string)
    value = audio.numpy()
    return value, rate

def get_encode_wav_compressed(tensor, sample_rate):
    # Defined in the `audio` function in `tensorboardX/summary.py`
    audio_string = encode_wav(tensor, sample_rate)
    audio, rate = tf.audio.decode_wav(audio_string)
    value = audio.numpy()
    print(value.shape)
    print(rate)
    return value, rate

@pytest.fixture
def prepare(testdir):
    # Ref: https://pytorch.org/docs/stable/tensorboard.html
    log_dir = os.path.join(testdir.tmpdir, 'run')

    x, rate = get_audio()
    writer = SummaryWriter(log_dir)
    writer.add_audio('my_audio', x, 0, sample_rate=rate)
    writer.close()

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
    x, rate = get_audio()
    x_compressed, rate_compressed = get_wave_compressed(x, rate)
    assert rate == rate_compressed
    reader = SummaryReader(tmpinfo["event_file"], pivot=True)
    # Test raw functions
    # - Test `raw_tags` and `get_raw_tags`
    assert reader.raw_tags == reader.get_raw_tags()
    assert reader.raw_tags['audio'] == reader.get_raw_tags('audio')
    assert set(reader.raw_tags['audio']) == {'my_audio'}
    # - Test `raw_events` and `get_raw_events`
    assert reader.raw_events == reader.get_raw_events()
    assert reader.raw_events['audio'] == reader.get_raw_events('audio')
    assert reader.raw_events['audio']['my_audio'] == reader.get_raw_events('audio', 'my_audio')
    # - (my_audio) Test raw event count & type
    events = reader.get_raw_events('audio', 'my_audio')
    assert len(events) == 1
    e: AudioEvent = events[0]
    assert type(e) == AudioEvent
    audio, sample_rate = tf.audio.decode_wav(e.encoded_audio_string)
    value = audio.numpy()
    assert rate == sample_rate
    assert (e.step, value.tolist()) == (0, x_compressed.tolist())
    assert e.content_type == 'audio/wav'
    assert e.length_frames == len(x_compressed)
    assert e.sample_rate == rate

def check_others(reader):
    assert len(reader.scalars) == 0
    assert len(reader.tensors) == 0
    assert len(reader.histograms) == 0
    assert len(reader.images) == 0
    assert len(reader.hparams) == 0
    assert len(reader.text) == 0

def test_log_dir(prepare, testdir):
    tmpinfo = get_tmpdir_info(testdir.tmpdir)
    x, rate = get_audio()
    x_compressed, rate_compressed = get_wave_compressed(x, rate)
    assert rate == rate_compressed
    # Test pivot
    reader = SummaryReader(tmpinfo["log_dir"], pivot=True, extra_columns={
                           'dir_name', 'file_name'})
    assert len(reader.children) == 1
    df = reader.audio
    assert df.columns.to_list() == ['step', 'my_audio', 'dir_name', 'file_name']
    assert df['step'].to_list() == [0]
    assert df['my_audio'][0].tolist() == x_compressed.tolist()
    assert df['dir_name'].to_list() == ['']
    assert df['file_name'].to_list() == [tmpinfo["event_filename"]]
    check_others(reader)
    # Test default
    reader = SummaryReader(tmpinfo["log_dir"], extra_columns={
                           'dir_name', 'file_name'})
    df = reader.audio
    assert len(reader.children) == 1
    assert df.columns.to_list() == ['step', 'tag', 'value', 'dir_name', 'file_name']
    assert df['step'].to_list() == [0]
    assert df['tag'].to_list() == ['my_audio']
    assert df['value'][0].tolist() == x_compressed.tolist()
    assert df['dir_name'].to_list() == ['']
    assert df['file_name'].to_list() == [tmpinfo["event_filename"]]
    check_others(reader)
    # Test pivot & all columns
    reader = SummaryReader(tmpinfo["log_dir"], pivot=True, extra_columns={
                           'content_type', 'length_frames', 'sample_rate', 'dir_name', 'file_name'})
    assert len(reader.children) == 1
    df = reader.audio
    assert df.columns.to_list() == ['step', 'my_audio', 'my_audio/content_type', 'my_audio/length_frames', 'my_audio/sample_rate', 'dir_name', 'file_name']
    assert df['step'].to_list() == [0]
    assert df['my_audio'][0].tolist() == x_compressed.tolist()
    assert df['my_audio/content_type'][0] == 'audio/wav'
    assert df['my_audio/length_frames'][0] == x.shape[1]
    assert df['my_audio/sample_rate'][0] == rate
    assert df['dir_name'].to_list() == ['']
    assert df['file_name'].to_list() == [tmpinfo["event_filename"]]
    check_others(reader)
    # Test all columns
    reader = SummaryReader(tmpinfo["log_dir"], extra_columns={
                           'content_type', 'length_frames', 'sample_rate', 'dir_name', 'file_name'})
    df = reader.audio
    assert len(reader.children) == 1
    assert df.columns.to_list() == ['step', 'tag', 'value', 'content_type', 'length_frames', 'sample_rate', 'dir_name', 'file_name']
    assert df['step'].to_list() == [0]
    assert df['tag'].to_list() == ['my_audio']
    assert df['value'][0].tolist() == x_compressed.tolist()
    assert df['content_type'][0] == 'audio/wav'
    assert df['length_frames'][0] == x.shape[1]
    assert df['sample_rate'][0] == rate
    assert df['dir_name'].to_list() == ['']
    assert df['file_name'].to_list() == [tmpinfo["event_filename"]]
    check_others(reader)

# TODO: tensorflow: multiple audio (batch size > 1)
# TODO: multiple channels

# TODO: tensor_to_audio: tensor[0][1]?
# TODO: tensor_to_audio: tensor[1]?
