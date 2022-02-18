import os
import tempfile

import pytest
import tensorboardX
import tensorflow as tf
from tbparse import SummaryReader
from tensorboard.backend.event_processing.event_accumulator import TensorEvent
from torch.utils.tensorboard import SummaryWriter


@pytest.fixture
def prepare(testdir):
    # Ref: https://pytorch.org/docs/stable/tensorboard.html
    # Ref: https://github.com/WarrenWeckesser/wavio
    log_dir = os.path.join(testdir.tmpdir, 'run')

    writer = SummaryWriter(log_dir)
    writer.add_text('textA', 'lorem ipsum', 0)
    writer.add_text('textA', 'dolor sit amet', 1)
    writer.add_text('textB', 'consectetur adipiscing', 0)
    writer.add_text('textB', 'elit', 1)
    writer.close()

def test_tensorboardX(prepare, testdir):
    # Note: tensorboardX automatically escapes special characters.
    # Prepare Log
    log_dir_th = os.path.join(testdir.tmpdir, 'run')
    tmpdir_tbx = tempfile.TemporaryDirectory()
    log_dir_tbx = os.path.join(tmpdir_tbx.name, 'run')
    writer = tensorboardX.SummaryWriter(log_dir_tbx)
    writer.add_text('textA', 'lorem ipsum', 0)
    writer.add_text('textA', 'dolor sit amet', 1)
    writer.add_text('textB', 'consectetur adipiscing', 0)
    writer.add_text('textB', 'elit', 1)
    writer.close()
    # (default) Parse & Compare
    df_th = SummaryReader(log_dir_th).tensors
    df_tbx = SummaryReader(log_dir_tbx).tensors
    assert df_th.equals(df_tbx)
    df_th = SummaryReader(log_dir_th).text
    df_tbx = SummaryReader(log_dir_tbx).text
    assert df_th.equals(df_tbx)
    # (pivot) Parse & Compare
    df_th = SummaryReader(log_dir_th, pivot=True).tensors
    df_tbx = SummaryReader(log_dir_tbx, pivot=True).tensors
    assert df_th.equals(df_tbx)
    df_th = SummaryReader(log_dir_th, pivot=True).text
    df_tbx = SummaryReader(log_dir_tbx, pivot=True).text
    assert df_th.equals(df_tbx)

def test_tensorflow(prepare, testdir):
    # Prepare Log
    log_dir_th = os.path.join(testdir.tmpdir, 'run')
    tmpdir_tf = tempfile.TemporaryDirectory()
    log_dir_tf = os.path.join(tmpdir_tf.name, 'run')
    writer = tf.summary.create_file_writer(log_dir_tf)
    writer.set_as_default()
    tf.summary.text('textA', 'lorem ipsum', 0)
    tf.summary.text('textA', 'dolor sit amet', 1)
    tf.summary.text('textB', 'consectetur adipiscing', 0)
    tf.summary.text('textB', 'elit', 1)
    writer.close()
    # (default) Parse & Compare
    df_th = SummaryReader(log_dir_th).tensors
    df_tf = SummaryReader(log_dir_tf).tensors
    assert df_th.equals(df_tf)
    df_th = SummaryReader(log_dir_th).text
    df_tf = SummaryReader(log_dir_tf).text
    assert df_th.equals(df_tf)
    # (pivot) Parse & Compare
    df_th = SummaryReader(log_dir_th, pivot=True).tensors
    df_tf = SummaryReader(log_dir_tf, pivot=True).tensors
    assert df_th.equals(df_tf)
    df_th = SummaryReader(log_dir_th, pivot=True).text
    df_tf = SummaryReader(log_dir_tf, pivot=True).text
    assert df_th.equals(df_tf)

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
    reader = SummaryReader(tmpinfo["event_file"], pivot=True)
    # Test raw functions
    # - Test `raw_tags` and `get_raw_tags`
    assert reader.raw_tags == reader.get_raw_tags()
    assert reader.raw_tags['text'] == reader.get_raw_tags('text')
    assert set(reader.raw_tags['text']) == {'textA/text_summary', 'textB/text_summary'}
    # - Test `raw_events` and `get_raw_events`
    assert reader.raw_events == reader.get_raw_events()
    assert reader.raw_events['text'] == reader.get_raw_events('text')
    assert reader.raw_events['text']['textA/text_summary'] == reader.get_raw_events('text', 'textA/text_summary')
    assert reader.raw_events['text']['textB/text_summary'] == reader.get_raw_events('text', 'textB/text_summary')
    # - (textA) Test raw event count & type
    events = reader.get_raw_events('text', 'textA/text_summary')
    assert len(events) == 2
    assert type(events[0]) == TensorEvent
    value = tf.make_ndarray(events[0].tensor_proto).item().decode('utf-8')
    assert (events[0].step, value) == (0, "lorem ipsum")
    value = tf.make_ndarray(events[1].tensor_proto).item().decode('utf-8')
    assert (events[1].step, value) == (1, "dolor sit amet")
    # - (textB) Test raw event count & type
    events = reader.get_raw_events('text', 'textB/text_summary')
    assert len(events) == 2
    assert type(events[0]) == TensorEvent
    value = tf.make_ndarray(events[0].tensor_proto).item().decode('utf-8')
    assert (events[0].step, value) == (0, "consectetur adipiscing")
    value = tf.make_ndarray(events[1].tensor_proto).item().decode('utf-8')
    assert (events[1].step, value) == (1, "elit")

def check_others(reader):
    assert len(reader.scalars) == 0
    assert len(reader.tensors) == 0
    assert len(reader.histograms) == 0
    assert len(reader.hparams) == 0

def test_log_dir(prepare, testdir):
    tmpinfo = get_tmpdir_info(testdir.tmpdir)
    # Test pivot
    reader = SummaryReader(tmpinfo["log_dir"], pivot=True, extra_columns={
                           'dir_name', 'file_name'})
    assert len(reader.children) == 1
    assert reader.text.columns.to_list() == ['step', 'textA', 'textB', 'dir_name', 'file_name']
    assert reader.text['step'].to_list() == [0, 1]
    assert reader.text['textA'].to_list() == ["lorem ipsum", "dolor sit amet"]
    assert reader.text['textB'].to_list() == ["consectetur adipiscing", "elit"]
    assert reader.text['dir_name'].to_list() == [''] * 2
    assert reader.text['file_name'].to_list() == [tmpinfo["event_filename"]] * 2
    check_others(reader)
    # Test default
    reader = SummaryReader(tmpinfo["log_dir"], extra_columns={
                           'dir_name', 'file_name'})
    assert reader.text.columns.to_list() == ['step', 'tag', 'value', 'dir_name', 'file_name']
    assert reader.text['step'].to_list() == [0, 1] * 2
    assert reader.text['tag'].to_list() == ['textA'] * 2 + ['textB'] * 2
    assert reader.text['value'].to_list() == \
        ["lorem ipsum", "dolor sit amet", "consectetur adipiscing", "elit"]
    assert reader.text['dir_name'].to_list() == [''] * 4
    assert reader.text['file_name'].to_list() == [tmpinfo["event_filename"]] * 4
    check_others(reader)
