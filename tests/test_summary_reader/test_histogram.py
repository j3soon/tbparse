import os
import re
from typing import List
import pandas as pd
import pytest
import tempfile
import numpy as np
from numpy.testing import assert_almost_equal
import tensorflow as tf
from tensorboard.backend.event_processing.event_accumulator import HistogramEvent, ScalarEvent
from tbparse import SummaryReader
from torch.utils.tensorboard import SummaryWriter
import tensorboardX

RND_STATE = 1234
N_EVENTS = 10
N_PARTICLES = 1000
MU = 0
SIGMA = 2

@pytest.fixture
def prepare(testdir):
    # Ref: https://pytorch.org/docs/stable/tensorboard.html
    log_dir = os.path.join(testdir.tmpdir, 'run')
    writer = SummaryWriter(log_dir)
    rng = np.random.RandomState(RND_STATE)
    for i in range(N_EVENTS):
        x = rng.normal(MU, SIGMA, size=N_PARTICLES)
        writer.add_histogram('dist', x + i, i)
    writer.close()
    """
    run
    └── events.out.tfevents.<id-1>
    """

def test_tensorflow(prepare, testdir):
    # Note: tensorflow's `histograms` are actually logged as tensors.
    #       Therefore, we need to convert them manually.
    # Prepare Log
    log_dir_th = os.path.join(testdir.tmpdir, 'run')
    tmpdir = tempfile.TemporaryDirectory()
    log_dir_tf = os.path.join(tmpdir.name, 'run')
    writer = tf.summary.create_file_writer(log_dir_tf)
    writer.set_as_default()
    rng = np.random.RandomState(RND_STATE)
    for i in range(N_EVENTS):
        x = rng.normal(MU, SIGMA, size=N_PARTICLES)
        tf.summary.histogram('dist', x + i, i)
    writer.close()
    # Test default
    df_th = SummaryReader(log_dir_th).histograms
    df_tf = SummaryReader(log_dir_tf).tensors
    for i in range(N_EVENTS):
        hist_dict = SummaryReader.buckets_to_histogram_dict(df_tf['dist'][i])
        assert len(hist_dict['counts']) + 1 == \
            len(hist_dict['limits'])
        assert sum(hist_dict['counts']) == N_PARTICLES
    df_th.drop(columns=['dist/limits', 'dist/counts'], inplace=True)
    df_tf.drop(columns=['dist'], inplace=True)
    assert(df_th.equals(df_tf))

def test_tensorboardX(prepare, testdir):
    # Prepare Log
    log_dir_th = os.path.join(testdir.tmpdir, 'run')
    tmpdir = tempfile.TemporaryDirectory()
    log_dir_tbx = os.path.join(tmpdir.name, 'run')
    writer = tensorboardX.SummaryWriter(log_dir_tbx)
    rng = np.random.RandomState(RND_STATE)
    for i in range(N_EVENTS):
        x = rng.normal(MU, SIGMA, size=N_PARTICLES)
        writer.add_histogram('dist', x + i, i)
    writer.close()
    # Test default
    df_th = SummaryReader(log_dir_th).histograms
    df_tbx = SummaryReader(log_dir_tbx).histograms
    assert(df_th.equals(df_tbx))
    # Test columns without tag
    columns = {'min', 'max', 'num', 'sum', 'sum_squares', 'dir_name'}
    df_th = SummaryReader(log_dir_th, columns=columns).histograms
    df_tbx = SummaryReader(log_dir_tbx, columns=columns).histograms
    assert(df_th.equals(df_tbx))
    # Test columns with tag
    columns = {'tag', 'min', 'max', 'num', 'sum', 'sum_squares', 'dir_name'}
    df_th = SummaryReader(log_dir_th, columns=columns).histograms
    df_tbx = SummaryReader(log_dir_tbx, columns=columns).histograms
    assert(df_th.equals(df_tbx))

def test_tensorboardX_hist_raw(prepare, testdir):
    # Note: log raw histogram may have different limits and counts
    # Prepare Log
    tmpdir_th = tempfile.TemporaryDirectory()
    log_dir_th = os.path.join(tmpdir_th.name, 'run')
    tmpdir_tbx = tempfile.TemporaryDirectory()
    log_dir_tbx = os.path.join(tmpdir_tbx.name, 'run')
    writer_th = tensorboardX.SummaryWriter(log_dir_th)
    writer_tbx = tensorboardX.SummaryWriter(log_dir_tbx)
    rng = np.random.RandomState(RND_STATE)
    for i in range(N_EVENTS):
        x = rng.normal(MU, SIGMA, size=N_PARTICLES)
        values = x + i
        counts, limits = np.histogram(values)
        sum_sq = values.dot(values)
        raw_dict = {
            'tag': 'dist',
            'min': values.min(),
            'max': values.max(),
            'num': len(values),
            'sum': values.sum(),
            'sum_squares': sum_sq,
            'bucket_limits': limits[1:].tolist(),
            'bucket_counts': counts.tolist(),
            'global_step': i,
        }
        writer_th.add_histogram_raw(**raw_dict)
        writer_tbx.add_histogram_raw(**raw_dict)
    writer_th.close()
    writer_tbx.close()
    # Test default
    df_th = SummaryReader(log_dir_th).histograms
    df_tbx = SummaryReader(log_dir_tbx).histograms
    assert(df_th.equals(df_tbx))
    # Test columns without tag
    columns = {'min', 'max', 'num', 'sum', 'sum_squares', 'dir_name'}
    df_th = SummaryReader(log_dir_th, columns=columns).histograms
    df_tbx = SummaryReader(log_dir_tbx, columns=columns).histograms
    assert(df_th.equals(df_tbx))
    # Test columns with tag
    columns = {'tag', 'min', 'max', 'num', 'sum', 'sum_squares', 'dir_name'}
    df_th = SummaryReader(log_dir_th, columns=columns).histograms
    df_tbx = SummaryReader(log_dir_tbx, columns=columns).histograms
    assert(df_th.equals(df_tbx))

def test_event_file_raw(prepare, testdir):
    log_dir = os.path.join(testdir.tmpdir, 'run')
    dirs = os.listdir(log_dir)
    assert len(dirs) == 1
    event_file = os.path.join(log_dir, dirs[0])
    reader = SummaryReader(event_file)
    # Test raw functions
    # - Test `raw_tags` and `get_raw_tags`
    assert reader.raw_tags == reader.get_raw_tags()
    assert reader.raw_tags['histograms'] == reader.get_raw_tags('histograms')
    assert set(reader.raw_tags['histograms']) == {'dist'}
    # - Test `raw_events` and `get_raw_events`
    assert reader.raw_events == reader.get_raw_events()
    assert reader.raw_events['histograms'] == reader.get_raw_events('histograms')
    assert reader.raw_events['histograms']['dist'] == reader.get_raw_events(
        'histograms', 'dist')
    # - Test raw event count & type
    events: List[HistogramEvent] = reader.get_raw_events('histograms', 'dist')
    assert len(events) == N_EVENTS
    assert type(events[0]) == HistogramEvent
    for i in range(N_EVENTS):
        assert events[i].step == i
        assert events[i].histogram_value.num == N_PARTICLES

def test_log_dir(prepare, testdir):
    log_dir = os.path.join(testdir.tmpdir, 'run')
    # Test default columns
    reader = SummaryReader(log_dir)
    assert len(reader.children) == 1
    df = SummaryReader(log_dir).histograms
    assert df.columns.to_list(
    ) == ['step', 'dist/limits', 'dist/counts']
    assert df['step'].to_list() == [i for i in range(N_EVENTS)]
    for i in range(N_EVENTS):
        assert len(df['dist/counts'][i]) + 1 == \
            len(df['dist/limits'][i])
        assert sum(df['dist/counts'][i]) == N_PARTICLES
    # Test tag columns
    df = SummaryReader(log_dir, columns={'tag'}).histograms
    assert df.columns.to_list(
    ) == ['step', 'tag', 'limits', 'counts']
    assert df['step'].to_list() == [i for i in range(N_EVENTS)]
    assert df['tag'].to_list() == [
        'dist' for _ in range(N_EVENTS)]
    for i in range(N_EVENTS):
        assert len(df['counts'][i]) + 1 == \
            len(df['limits'][i])
        assert sum(df['counts'][i]) == N_PARTICLES
    # Test all columns without tag
    df = SummaryReader(log_dir, columns={
                           'min', 'max', 'num', 'sum', 'sum_squares',
                           'wall_time', 'dir_name', 'file_name'}).histograms
    assert df.columns.to_list(
        ) == ['step', 'dist/limits', 'dist/counts', 'dist/min', 'dist/max',
              'dist/num', 'dist/sum', 'dist/sum_squares', 'wall_time',
              'dir_name', 'file_name']
    assert df['step'].to_list() == [i for i in range(N_EVENTS)]
    for i in range(N_EVENTS):
        assert len(df['dist/counts'][i]) + 1 == \
            len(df['dist/limits'][i])
        assert sum(df['dist/counts'][i]) == N_PARTICLES
        assert np.isscalar(df['dist/min'][i])
        assert np.isscalar(df['dist/max'][i])
        assert df['dist/num'][i] == N_PARTICLES
        assert np.isscalar(df['dist/sum'][i])
        assert np.isscalar(df['dist/sum_squares'][i])
    # Test all columns
    df = SummaryReader(log_dir, columns={
                           'tag', 'min', 'max', 'num', 'sum', 'sum_squares',
                           'wall_time', 'dir_name', 'file_name'}).histograms
    assert df.columns.to_list(
        ) == ['step', 'tag', 'limits', 'counts', 'min', 'max', 'num', 'sum',
        'sum_squares', 'wall_time', 'dir_name', 'file_name']
    assert df['step'].to_list() == [i for i in range(N_EVENTS)]
    assert df['tag'].to_list() == ['dist' for _ in range(N_EVENTS)]
    for i in range(N_EVENTS):
        assert len(df['counts'][i]) + 1 == \
            len(df['limits'][i])
        assert sum(df['counts'][i]) == N_PARTICLES
        assert np.isscalar(df['min'][i])
        assert np.isscalar(df['max'][i])
        assert df['num'][i] == N_PARTICLES
        assert np.isscalar(df['sum'][i])
        assert np.isscalar(df['sum_squares'][i])

def test_histogram_to_cdf():
    counts = [1, 3]
    limits = [-10, 0, 10]
    x = [-20, -11, -10, -9, -1, 0, 1, 9, 10, 11, 20]
    y = SummaryReader.histogram_to_cdf(counts, limits, x)
    y = list(y * np.sum(counts))
    expected_y = [0., 0., 0., 0.1, 0.9, 1, 1+3*0.1, 1+3*0.9, 4., 4., 4.]
    assert y == expected_y