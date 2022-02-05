import os
import re
import tempfile
from typing import List

import pytest
import tensorboardX
import tensorflow as tf
from tbparse import SummaryReader
from tensorboard.backend.event_processing.event_accumulator import (
    ScalarEvent, TensorEvent)
from torch.utils.tensorboard import SummaryWriter

N_RUNS = 3
N_EVENTS = 5

@pytest.fixture
def prepare(testdir):
    # Use torch for main tests, logs for tensorboard and tensorboardX are
    # generated in their own tests.
    # Ref: https://pytorch.org/docs/stable/tensorboard.html
    log_dir = os.path.join(testdir.tmpdir, 'run')
    for i in range(N_RUNS):
        writer = SummaryWriter(os.path.join(log_dir, f'run{i}'))
        for j in range(N_EVENTS):
            writer.add_scalar('y=2x+C', j * 2 + i, j, new_style=True)
            writer.add_scalar('y=3x+C', j * 3 + i, j, new_style=True)
        writer.close()
    """
    run
    ├── run0
    │   └── events.out.tfevents.<id-1>
    ├── run1
    │   └── events.out.tfevents.<id-2>
    └── run2
        └── events.out.tfevents.<id-3>
    """

def test_tensorboardX(prepare, testdir):
    pass
    # Note: tensorboardX doesn't support logging tensors.

def test_tensorflow(prepare, testdir):
    # Prepare Log
    log_dir_th = os.path.join(testdir.tmpdir, 'run')
    tmpdir_tf = tempfile.TemporaryDirectory()
    log_dir_tf = os.path.join(tmpdir_tf.name, 'run')
    for i in range(N_RUNS):
        writer = tf.summary.create_file_writer(os.path.join(log_dir_tf, f'run{i}'))
        writer.set_as_default()
        for j in range(N_EVENTS):
            tf.summary.scalar('y=2x+C', j * 2 + i, j)
            tf.summary.scalar('y=3x+C', j * 3 + i, j)
        writer.close()
    # (default) Parse & Compare
    df_th = SummaryReader(log_dir_th).tensors
    df_tf = SummaryReader(log_dir_tf).tensors
    assert(df_th.equals(df_tf))
    # (dir_name) Parse & Compare
    df_th = SummaryReader(log_dir_th, extra_columns={'dir_name'}).tensors
    df_tf = SummaryReader(log_dir_tf, extra_columns={'dir_name'}).tensors
    assert(df_th.equals(df_tf))
    # (pivot) Parse & Compare
    df_th = SummaryReader(log_dir_th, pivot=True).tensors
    df_tf = SummaryReader(log_dir_tf, pivot=True).tensors
    assert(df_th.equals(df_tf))
    # (pivot & dir_name) Parse & Compare
    df_th = SummaryReader(log_dir_th, pivot=True, extra_columns={'dir_name'}).tensors
    df_tf = SummaryReader(log_dir_tf, pivot=True, extra_columns={'dir_name'}).tensors
    assert(df_th.equals(df_tf))

def test_event_file_raw(prepare, testdir):
    log_dir = os.path.join(testdir.tmpdir, 'run')
    run_dir = os.path.join(log_dir, 'run0')
    dirs = os.listdir(run_dir)
    assert len(dirs) == 1
    event_file = os.path.join(run_dir, dirs[0])
    reader = SummaryReader(event_file, pivot=True)
    # Test raw functions
    # - Test `raw_tags` and `get_raw_tags`
    assert reader.raw_tags == reader.get_raw_tags()
    assert reader.raw_tags['tensors'] == reader.get_raw_tags('tensors')
    assert set(reader.raw_tags['tensors']) == {'y=2x+C', 'y=3x+C'}
    # - Test `raw_events` and `get_raw_events`
    assert reader.raw_events == reader.get_raw_events()
    assert reader.raw_events['tensors'] == reader.get_raw_events('tensors')
    assert reader.raw_events['tensors']['y=2x+C'] == reader.get_raw_events('tensors', 'y=2x+C')
    # - Test raw event count & type
    events: List[TensorEvent] = reader.get_raw_events('tensors', 'y=2x+C')
    assert len(events) == N_EVENTS
    assert type(events[0]) == TensorEvent
    for i in range(N_EVENTS):
        value = tf.make_ndarray(events[i].tensor_proto)
        assert (events[i].step, value) == (i, i * 2)

def check_others(reader):
    assert len(reader.scalars) == 0
    assert len(reader.histograms) == 0
    assert len(reader.hparams) == 0

def test_event_file(prepare, testdir):
    log_dir = os.path.join(testdir.tmpdir, 'run')
    run_dir = os.path.join(log_dir, 'run0')
    dirs = os.listdir(run_dir)
    assert len(dirs) == 1
    event_filename = dirs[0]
    event_file = os.path.join(run_dir, event_filename)
    # Test pivot
    reader = SummaryReader(event_file, pivot=True)
    assert reader.tensors.columns.to_list() == ['step', 'y=2x+C', 'y=3x+C']
    assert reader.tensors['step'].to_list() == [i for i in range(N_EVENTS)]
    assert reader.tensors['y=2x+C'].to_list() == [i * 2 for i in range(N_EVENTS)]
    assert reader.tensors['y=3x+C'].to_list() == [i * 3 for i in range(N_EVENTS)]
    check_others(reader)
    # Test additional tag column
    reader = SummaryReader(event_file)
    assert reader.tensors.columns.to_list() == ['step', 'tag', 'value']
    assert reader.tensors['step'].to_list()[:N_EVENTS] == [i for i in range(N_EVENTS)]
    assert reader.tensors['step'].to_list()[N_EVENTS:] == [i for i in range(N_EVENTS)]
    assert reader.tensors['tag'].to_list()[:N_EVENTS] == ['y=2x+C'] * N_EVENTS
    assert reader.tensors['tag'].to_list()[N_EVENTS:] == ['y=3x+C'] * N_EVENTS
    assert reader.tensors['value'].to_list()[:N_EVENTS] == [i * 2 for i in range(N_EVENTS)]
    assert reader.tensors['value'].to_list()[N_EVENTS:] == [i * 3 for i in range(N_EVENTS)]
    check_others(reader)
    # Test pivot & additional wall_time column
    reader = SummaryReader(event_file, pivot=True, extra_columns={'wall_time'})
    assert reader.tensors.columns.to_list() == ['step', 'y=2x+C', 'y=3x+C', 'wall_time']
    assert len(reader.tensors['wall_time']) == N_EVENTS
    check_others(reader)
    # Test pivot & additional dir_name column
    reader = SummaryReader(event_file, pivot=True, extra_columns={'dir_name'})
    assert reader.tensors.columns.to_list() == ['step', 'y=2x+C', 'y=3x+C', 'dir_name']
    assert reader.tensors['dir_name'].to_list() == [''] * N_EVENTS
    check_others(reader)
    # Test pivot & additional file_name column
    reader = SummaryReader(event_file, pivot=True, extra_columns={'file_name'})
    assert reader.tensors.columns.to_list() == ['step', 'y=2x+C', 'y=3x+C', 'file_name']
    assert reader.tensors['file_name'].to_list() == [event_filename] * N_EVENTS
    check_others(reader)
    # Test all columns
    reader = SummaryReader(event_file, extra_columns={
                           'wall_time', 'dir_name', 'file_name'})
    assert reader.tensors.columns.to_list() == ['step', 'tag', 'value', 'wall_time', 'dir_name', 'file_name']
    assert reader.tensors['step'].to_list()[:N_EVENTS] == [i for i in range(N_EVENTS)]
    assert reader.tensors['step'].to_list()[N_EVENTS:] == [i for i in range(N_EVENTS)]
    assert reader.tensors['tag'].to_list()[:N_EVENTS] == ['y=2x+C'] * N_EVENTS
    assert reader.tensors['tag'].to_list()[N_EVENTS:] == ['y=3x+C'] * N_EVENTS
    assert reader.tensors['value'].to_list()[:N_EVENTS] == [i * 2 for i in range(N_EVENTS)]
    assert reader.tensors['value'].to_list()[N_EVENTS:] == [i * 3 for i in range(N_EVENTS)]
    assert len(reader.tensors['wall_time']) == N_EVENTS * 2
    assert reader.tensors['dir_name'].to_list() == [''] * (N_EVENTS * 2)
    assert reader.tensors['file_name'].to_list() == [event_filename] * (N_EVENTS * 2)
    check_others(reader)

def test_run_dir(prepare, testdir):
    log_dir = os.path.join(testdir.tmpdir, 'run')
    run_dir = os.path.join(log_dir, 'run0')
    dirs = os.listdir(run_dir)
    assert len(dirs) == 1
    event_filename = dirs[0]
    # Test pivot
    reader = SummaryReader(run_dir, pivot=True, extra_columns={
                           'wall_time', 'dir_name', 'file_name'})
    assert len(reader.children) == 1
    assert reader.tensors.columns.to_list() == ['step', 'y=2x+C', 'y=3x+C', 'wall_time', 'dir_name', 'file_name']
    assert reader.tensors['step'].to_list() == [i for i in range(N_EVENTS)]
    assert reader.tensors['y=2x+C'].to_list() == [i * 2 for i in range(N_EVENTS)]
    assert reader.tensors['y=3x+C'].to_list() == [i * 3 for i in range(N_EVENTS)]
    assert len(reader.tensors['wall_time']) == N_EVENTS
    assert len(reader.tensors['wall_time'][0]) == 2
    assert reader.tensors['dir_name'].to_list() == [''] * N_EVENTS
    assert reader.tensors['file_name'].to_list() == [event_filename] * N_EVENTS
    check_others(reader)
    # Test all columns
    reader = SummaryReader(run_dir, extra_columns={
                           'wall_time', 'dir_name', 'file_name'})
    assert reader.tensors.columns.to_list() == ['step', 'tag', 'value', 'wall_time', 'dir_name', 'file_name']
    assert reader.tensors['step'].to_list()[:N_EVENTS] == [i for i in range(N_EVENTS)]
    assert reader.tensors['step'].to_list()[N_EVENTS:] == [i for i in range(N_EVENTS)]
    assert reader.tensors['tag'].to_list()[:N_EVENTS] == ['y=2x+C'] * N_EVENTS
    assert reader.tensors['tag'].to_list()[N_EVENTS:] == ['y=3x+C'] * N_EVENTS
    assert reader.tensors['value'].to_list()[:N_EVENTS] == [i * 2 for i in range(N_EVENTS)]
    assert reader.tensors['value'].to_list()[N_EVENTS:] == [i * 3 for i in range(N_EVENTS)]
    assert len(reader.tensors['wall_time']) == N_EVENTS * 2
    assert reader.tensors['dir_name'].to_list() == [''] * (N_EVENTS * 2)
    assert reader.tensors['file_name'].to_list() == [event_filename] * (N_EVENTS * 2)
    check_others(reader)

def test_log_dir(prepare, testdir):
    log_dir = os.path.join(testdir.tmpdir, 'run')
    # Test pivot
    reader = SummaryReader(log_dir, pivot=True, extra_columns={
                           'dir_name', 'file_name'})
    assert len(reader.children) == N_RUNS
    assert reader.tensors.columns.to_list() == ['step', 'y=2x+C', 'y=3x+C', 'dir_name', 'file_name']
    for i in range(N_RUNS):
        run_dir = os.path.join(log_dir, f'run{i}')
        dirs = os.listdir(run_dir)
        assert len(dirs) == 1
        event_filename = dirs[0]
        s, e = i*N_EVENTS, (i+1)*N_EVENTS
        assert reader.tensors['step'][s:e].to_list() == [j for j in range(N_EVENTS)]
        assert reader.tensors['y=2x+C'][s:e].to_list() == [j * 2 + i for j in range(N_EVENTS)]
        assert reader.tensors['y=3x+C'][s:e].to_list() == [j * 3 + i for j in range(N_EVENTS)]
        assert reader.tensors['dir_name'][s:e].to_list() == [f'run{i}'] * N_EVENTS
        assert reader.tensors['file_name'][s:e].to_list() == [event_filename] * N_EVENTS
    check_others(reader)
    # Test all columns
    reader = SummaryReader(log_dir, extra_columns={
                           'wall_time', 'dir_name', 'file_name'})
    assert reader.tensors.columns.to_list() == ['step', 'tag', 'value', 'wall_time', 'dir_name', 'file_name']
    for i in range(N_RUNS):
        run_dir = os.path.join(log_dir, f'run{i}')
        dirs = os.listdir(run_dir)
        assert len(dirs) == 1
        event_filename = dirs[0]
        s1, e1 = i*(N_EVENTS*2), i*(N_EVENTS*2) + N_EVENTS
        s2, e2 = (i+1)*(N_EVENTS*2) - N_EVENTS, (i+1)*(N_EVENTS*2)
        assert reader.tensors['step'].to_list()[s1:e1] == [j for j in range(N_EVENTS)]
        assert reader.tensors['step'].to_list()[s2:e2] == [j for j in range(N_EVENTS)]
        assert reader.tensors['tag'].to_list()[s1:e1] == ['y=2x+C'] * N_EVENTS
        assert reader.tensors['tag'].to_list()[s2:e2] == ['y=3x+C'] * N_EVENTS
        assert reader.tensors['value'].to_list()[s1:e1] == [j * 2 + i for j in range(N_EVENTS)]
        assert reader.tensors['value'].to_list()[s2:e2] == [j * 3 + i for j in range(N_EVENTS)]
        assert len(reader.tensors['wall_time']) == N_RUNS * N_EVENTS * 2
        assert reader.tensors['dir_name'][s1:e2].to_list() == [f'run{i}'] * (N_EVENTS * 2)
        assert reader.tensors['file_name'][s1:e2].to_list() == [event_filename] * (N_EVENTS * 2)
    check_others(reader)
