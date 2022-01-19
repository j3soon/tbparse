import os
import re
from typing import List
import pytest
import tempfile
import tensorflow as tf
from tensorboard.backend.event_processing.event_accumulator import ScalarEvent
from tbparse import SummaryReader
from torch.utils.tensorboard import SummaryWriter
import tensorboardX

N_RUNS = 3
N_EVENTS = 5

@pytest.fixture
def prepare(testdir):
    # Ref: https://pytorch.org/docs/stable/tensorboard.html
    log_dir = os.path.join(testdir.tmpdir, 'run')
    for i in range(N_RUNS):
        writer = SummaryWriter(os.path.join(log_dir, f'run{i}'))
        # writer.add_hparams({'name': 'test', 'run id': i}, {}, run_name='.')
        for j in range(N_EVENTS):
            writer.add_scalar('y=2x+C', j * 2 + i, j)
            writer.add_scalar('y=3x+C', j * 3 + i, j)
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

def test_tensorflow(prepare, testdir):
    # Note: tensorflow's `scalars` are actually logged as tensors.
    #       thus it is tested in `test_tensor.py`.
    # Prepare Log
    tmpdir = tempfile.TemporaryDirectory()
    log_dir_tf = os.path.join(tmpdir.name, 'run')
    for i in range(N_RUNS):
        writer = tf.summary.create_file_writer(os.path.join(log_dir_tf, f'run{i}'))
        writer.set_as_default()
        for j in range(N_EVENTS):
            tf.summary.scalar('y=2x+C', j * 2 + i, j)
            tf.summary.scalar('y=3x+C', j * 3 + i, j)
        writer.close()
    # (default) Parse & Compare
    df_tf = SummaryReader(log_dir_tf).scalars
    assert df_tf.empty

def test_tensorboardX(prepare, testdir):
    # Note: tensorboardX automatically escapes special characters.
    # Prepare Log
    log_dir_th = os.path.join(testdir.tmpdir, 'run')
    tmpdir = tempfile.TemporaryDirectory()
    log_dir_tbx = os.path.join(tmpdir.name, 'run')
    for i in range(N_RUNS):
        writer = tensorboardX.SummaryWriter(os.path.join(log_dir_tbx, f'run{i}'))
        for j in range(N_EVENTS):
            writer.add_scalar('y=2x+C', j * 2 + i, j)
            writer.add_scalar('y=3x+C', j * 3 + i, j)
        writer.close()
    # (default) Parse & Compare
    df_th = SummaryReader(log_dir_th).scalars
    df_tbx = SummaryReader(log_dir_tbx).scalars
    assert df_th['step'].to_list() == df_tbx['step'].to_list()
    assert df_th['y=2x+C'].to_list() == df_tbx['y_2x_C'].to_list()
    assert df_th['y=3x+C'].to_list() == df_tbx['y_3x_C'].to_list()
    # (dir_name) Parse & Compare
    df_th = SummaryReader(log_dir_th, cols={'dir_name'}).scalars
    df_tbx = SummaryReader(log_dir_tbx, cols={'dir_name'}).scalars
    assert df_th['step'].to_list() == df_tbx['step'].to_list()
    assert df_th['y=2x+C'].to_list() == df_tbx['y_2x_C'].to_list()
    assert df_th['y=3x+C'].to_list() == df_tbx['y_3x_C'].to_list()
    assert df_th['dir_name'].to_list() == df_tbx['dir_name'].to_list()
    # (tag & dir_name) Parse & Compare
    df_th = SummaryReader(log_dir_th, cols={'tag', 'dir_name'}).scalars
    df_tbx = SummaryReader(log_dir_tbx, cols={'tag', 'dir_name'}).scalars
    assert df_th['step'].to_list() == df_tbx['step'].to_list()
    r = re.compile(r'=|\+')
    escaped_th_tag = [r.sub('_', x) for x in df_th['tag'].to_list()]
    assert escaped_th_tag == df_tbx['tag'].to_list()
    df_th.drop(columns=['tag'], inplace=True)
    df_tbx.drop(columns=['tag'], inplace=True)
    assert(df_th.equals(df_tbx))

def test_event_file_raw(prepare, testdir):
    log_dir = os.path.join(testdir.tmpdir, 'run')
    run_dir = os.path.join(log_dir, 'run0')
    dirs = os.listdir(run_dir)
    assert len(dirs) == 1
    event_file = os.path.join(run_dir, dirs[0])
    reader = SummaryReader(event_file)
    # Test raw functions
    # - Test `raw_tags` and `get_raw_tags`
    assert reader.raw_tags == reader.get_raw_tags()
    assert reader.raw_tags['scalars'] == reader.get_raw_tags('scalars')
    assert set(reader.raw_tags['scalars']) == {'y=2x+C', 'y=3x+C'}
    # - Test `raw_events` and `get_raw_events`
    assert reader.raw_events == reader.get_raw_events()
    assert reader.raw_events['scalars'] == reader.get_raw_events('scalars')
    assert reader.raw_events['scalars']['y=2x+C'] == reader.get_raw_events(
        'scalars', 'y=2x+C')
    # - Test raw event count & type
    events: List[ScalarEvent] = reader.get_raw_events('scalars', 'y=2x+C')
    assert len(events) == N_EVENTS
    assert type(events[0]) == ScalarEvent
    for i in range(N_EVENTS):
        assert (events[i].step, events[i].value) == (i, i * 2)

def test_event_file(prepare, testdir):
    log_dir = os.path.join(testdir.tmpdir, 'run')
    run_dir = os.path.join(log_dir, 'run0')
    dirs = os.listdir(run_dir)
    assert len(dirs) == 1
    event_filename = dirs[0]
    event_file = os.path.join(run_dir, event_filename)
    # Test default columns
    reader = SummaryReader(event_file)
    assert reader.scalars.columns.to_list() == ['step', 'y=2x+C', 'y=3x+C']
    assert reader.scalars['step'].to_list() == [i for i in range(N_EVENTS)]
    assert reader.scalars['y=2x+C'].to_list() == [i *
                                                  2 for i in range(N_EVENTS)]
    assert reader.scalars['y=3x+C'].to_list() == [i *
                                                  3 for i in range(N_EVENTS)]
    # Test additional tag column
    reader = SummaryReader(event_file, cols={'tag'})
    assert reader.scalars.columns.to_list() == ['step', 'tag', 'value']
    assert reader.scalars['step'].to_list()[:N_EVENTS] == [
        i for i in range(N_EVENTS)]
    assert reader.scalars['step'].to_list()[N_EVENTS:] == [
        i for i in range(N_EVENTS)]
    assert reader.scalars['tag'].to_list()[:N_EVENTS] == [
        'y=2x+C' for _ in range(N_EVENTS)]
    assert reader.scalars['tag'].to_list()[N_EVENTS:] == [
        'y=3x+C' for _ in range(N_EVENTS)]
    assert reader.scalars['value'].to_list()[:N_EVENTS] == [
        i * 2 for i in range(N_EVENTS)]
    assert reader.scalars['value'].to_list()[N_EVENTS:] == [
        i * 3 for i in range(N_EVENTS)]
    # Test additional wall_time column
    reader = SummaryReader(event_file, cols={'wall_time'})
    assert reader.scalars.columns.to_list(
    ) == ['step', 'y=2x+C', 'y=3x+C', 'wall_time']
    assert len(reader.scalars['wall_time']) == N_EVENTS
    # Test additional dir_name column
    reader = SummaryReader(event_file, cols={'dir_name'})
    assert reader.scalars.columns.to_list(
    ) == ['step', 'y=2x+C', 'y=3x+C', 'dir_name']
    assert reader.scalars['dir_name'].to_list() == [
        '' for _ in range(N_EVENTS)]
    # Test additional file_name column
    reader = SummaryReader(event_file, cols={'file_name'})
    assert reader.scalars.columns.to_list(
    ) == ['step', 'y=2x+C', 'y=3x+C', 'file_name']
    assert reader.scalars['file_name'].to_list(
    ) == [event_filename for _ in range(N_EVENTS)]
    # Test all columns
    reader = SummaryReader(event_file, cols={
                           'tag', 'wall_time', 'dir_name', 'file_name'})
    assert reader.scalars.columns.to_list(
    ) == ['step', 'tag', 'value', 'wall_time', 'dir_name', 'file_name']
    assert reader.scalars['step'].to_list()[:N_EVENTS] == [
        i for i in range(N_EVENTS)]
    assert reader.scalars['step'].to_list()[N_EVENTS:] == [
        i for i in range(N_EVENTS)]
    assert reader.scalars['tag'].to_list()[:N_EVENTS] == [
        'y=2x+C' for _ in range(N_EVENTS)]
    assert reader.scalars['tag'].to_list()[N_EVENTS:] == [
        'y=3x+C' for _ in range(N_EVENTS)]
    assert reader.scalars['value'].to_list()[:N_EVENTS] == [
        i * 2 for i in range(N_EVENTS)]
    assert reader.scalars['value'].to_list()[N_EVENTS:] == [
        i * 3 for i in range(N_EVENTS)]
    assert len(reader.scalars['wall_time']) == N_EVENTS * 2
    assert reader.scalars['dir_name'].to_list(
    ) == ['' for _ in range(N_EVENTS * 2)]
    assert reader.scalars['file_name'].to_list(
    ) == [event_filename for _ in range(N_EVENTS * 2)]

def test_run_dir(prepare, testdir):
    log_dir = os.path.join(testdir.tmpdir, 'run')
    run_dir = os.path.join(log_dir, 'run0')
    dirs = os.listdir(run_dir)
    assert len(dirs) == 1
    event_filename = dirs[0]
    # Test columns without tag
    reader = SummaryReader(run_dir, cols={
                           'wall_time', 'dir_name', 'file_name'})
    assert len(reader.children) == 1
    assert reader.scalars.columns.to_list(
    ) == ['step', 'y=2x+C', 'y=3x+C', 'wall_time', 'dir_name', 'file_name']
    assert reader.scalars['step'].to_list() == [i for i in range(N_EVENTS)]
    assert reader.scalars['y=2x+C'].to_list() == [i *
                                                  2 for i in range(N_EVENTS)]
    assert reader.scalars['y=3x+C'].to_list() == [i *
                                                  3 for i in range(N_EVENTS)]
    assert len(reader.scalars['wall_time']) == N_EVENTS
    assert len(reader.scalars['wall_time'][0]) == 2
    assert reader.scalars['dir_name'].to_list() == [
        '' for _ in range(N_EVENTS)]
    assert reader.scalars['file_name'].to_list(
    ) == [event_filename for _ in range(N_EVENTS)]
    # Test columns with tag
    reader = SummaryReader(run_dir, cols={
                           'tag', 'wall_time', 'dir_name', 'file_name'})
    assert reader.scalars.columns.to_list(
    ) == ['step', 'tag', 'value', 'wall_time', 'dir_name', 'file_name']
    assert reader.scalars['step'].to_list()[:N_EVENTS] == [
        i for i in range(N_EVENTS)]
    assert reader.scalars['step'].to_list()[N_EVENTS:] == [
        i for i in range(N_EVENTS)]
    assert reader.scalars['tag'].to_list()[:N_EVENTS] == [
        'y=2x+C' for _ in range(N_EVENTS)]
    assert reader.scalars['tag'].to_list()[N_EVENTS:] == [
        'y=3x+C' for _ in range(N_EVENTS)]
    assert reader.scalars['value'].to_list()[:N_EVENTS] == [
        i * 2 for i in range(N_EVENTS)]
    assert reader.scalars['value'].to_list()[N_EVENTS:] == [
        i * 3 for i in range(N_EVENTS)]
    assert len(reader.scalars['wall_time']) == N_EVENTS * 2
    assert reader.scalars['dir_name'].to_list(
    ) == ['' for _ in range(N_EVENTS * 2)]
    assert reader.scalars['file_name'].to_list(
    ) == [event_filename for _ in range(N_EVENTS * 2)]

def test_log_dir(prepare, testdir):
    log_dir = os.path.join(testdir.tmpdir, 'run')
    # Test basic columns
    reader = SummaryReader(log_dir, cols={
                           'dir_name', 'file_name'})
    assert len(reader.children) == N_RUNS
    assert reader.scalars.columns.to_list(
    ) == ['step', 'y=2x+C', 'y=3x+C', 'dir_name', 'file_name']
    for i in range(N_RUNS):
        run_dir = os.path.join(log_dir, f'run{i}')
        dirs = os.listdir(run_dir)
        assert len(dirs) == 1
        event_filename = dirs[0]
        s, e = i*N_EVENTS, (i+1)*N_EVENTS
        assert reader.scalars['step'][s:e].to_list() == \
            [j for j in range(N_EVENTS)]
        assert reader.scalars['y=2x+C'][s:e].to_list() == \
            [j * 2 + i for j in range(N_EVENTS)]
        assert reader.scalars['y=3x+C'][s:e].to_list() == \
            [j * 3 + i for j in range(N_EVENTS)]
        assert reader.scalars['dir_name'][s:e].to_list() == \
            [f'run{i}' for _ in range(N_EVENTS)]
        assert reader.scalars['file_name'][s:e].to_list() == \
            [event_filename for _ in range(N_EVENTS)]
    # Test all columns
    reader = SummaryReader(log_dir, cols={
                           'tag', 'wall_time', 'dir_name', 'file_name'})
    assert reader.scalars.columns.to_list(
    ) == ['step', 'tag', 'value', 'wall_time', 'dir_name', 'file_name']
    for i in range(N_RUNS):
        run_dir = os.path.join(log_dir, f'run{i}')
        dirs = os.listdir(run_dir)
        assert len(dirs) == 1
        event_filename = dirs[0]
        s1, e1 = i*(N_EVENTS*2), i*(N_EVENTS*2) + N_EVENTS
        s2, e2 = (i+1)*(N_EVENTS*2) - N_EVENTS, (i+1)*(N_EVENTS*2)
        assert reader.scalars['step'].to_list()[s1:e1] == [
            j for j in range(N_EVENTS)]
        assert reader.scalars['step'].to_list()[s2:e2] == [
            j for j in range(N_EVENTS)]
        assert reader.scalars['tag'].to_list()[s1:e1] == [
            'y=2x+C' for _ in range(N_EVENTS)]
        assert reader.scalars['tag'].to_list()[s2:e2] == [
            'y=3x+C' for _ in range(N_EVENTS)]
        assert reader.scalars['value'].to_list()[s1:e1] == [
            j * 2 + i for j in range(N_EVENTS)]
        assert reader.scalars['value'].to_list()[s2:e2] == [
            j * 3 + i for j in range(N_EVENTS)]
        assert len(reader.scalars['wall_time']) == N_RUNS * N_EVENTS * 2
        assert reader.scalars['dir_name'][s1:e2].to_list() == \
            [f'run{i}' for _ in range(N_EVENTS * 2)]
        assert reader.scalars['file_name'][s1:e2].to_list() == \
            [event_filename for _ in range(N_EVENTS * 2)]
