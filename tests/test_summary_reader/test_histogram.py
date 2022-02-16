import os
import tempfile
from typing import List

import numpy as np
import pytest
import tensorboardX
import tensorflow as tf
from tbparse import SummaryReader
from tensorboard.backend.event_processing.event_accumulator import \
    HistogramEvent
from torch.utils.tensorboard import SummaryWriter

RND_STATE = 1234
N_EVENTS = 10
N_PARTICLES = 1000
MU = 0
SIGMA = 2

@pytest.fixture
def prepare(testdir):
    # Use torch for main tests, logs for tensorboard and tensorboardX are
    # generated in their own tests.
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
    tmpdir_tf = tempfile.TemporaryDirectory()
    log_dir_tf = os.path.join(tmpdir_tf.name, 'run')
    writer = tf.summary.create_file_writer(log_dir_tf)
    writer.set_as_default()
    rng = np.random.RandomState(RND_STATE)
    for i in range(N_EVENTS):
        x = rng.normal(MU, SIGMA, size=N_PARTICLES)
        tf.summary.histogram('dist', x + i, i)
    writer.close()
    # Test pivot
    df_th = SummaryReader(log_dir_th, pivot=True).histograms
    df_tf = SummaryReader(log_dir_tf, pivot=True).tensors
    for i in range(N_EVENTS):
        hist_dict = SummaryReader.buckets_to_histogram_dict(df_tf['dist'][i])
        assert len(hist_dict['counts']) + 1 == len(hist_dict['limits'])
        assert sum(hist_dict['counts']) == N_PARTICLES
    df_th.drop(columns=['dist/limits', 'dist/counts'], inplace=True)
    df_tf.drop(columns=['dist'], inplace=True)
    assert(df_th.equals(df_tf))

def test_tensorboardX(prepare, testdir):
    # Prepare Log
    log_dir_th = os.path.join(testdir.tmpdir, 'run')
    tmpdir_tbx = tempfile.TemporaryDirectory()
    log_dir_tbx = os.path.join(tmpdir_tbx.name, 'run')
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
    # Test pivot
    df_th = SummaryReader(log_dir_th, pivot=True).histograms
    df_tbx = SummaryReader(log_dir_tbx, pivot=True).histograms
    assert(df_th.equals(df_tbx))
    # Test all columns
    columns = {'min', 'max', 'num', 'sum', 'sum_squares', 'dir_name'}
    df_th = SummaryReader(log_dir_th, extra_columns=columns).histograms
    df_tbx = SummaryReader(log_dir_tbx, extra_columns=columns).histograms
    assert(df_th.equals(df_tbx))
    # Test pivot & all columns
    columns = {'min', 'max', 'num', 'sum', 'sum_squares', 'dir_name'}
    df_th = SummaryReader(log_dir_th, pivot=True, extra_columns=columns).histograms
    df_tbx = SummaryReader(log_dir_tbx, pivot=True, extra_columns=columns).histograms
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
    # Test pivot
    df_th = SummaryReader(log_dir_th, pivot=True).histograms
    df_tbx = SummaryReader(log_dir_tbx, pivot=True).histograms
    assert(df_th.equals(df_tbx))
    # Test all columns
    columns = {'min', 'max', 'num', 'sum', 'sum_squares', 'dir_name'}
    df_th = SummaryReader(log_dir_th, extra_columns=columns).histograms
    df_tbx = SummaryReader(log_dir_tbx, extra_columns=columns).histograms
    assert(df_th.equals(df_tbx))
    # Test pivot & all columns
    columns = {'min', 'max', 'num', 'sum', 'sum_squares', 'dir_name'}
    df_th = SummaryReader(log_dir_th, pivot=True, extra_columns=columns).histograms
    df_tbx = SummaryReader(log_dir_tbx, pivot=True, extra_columns=columns).histograms
    assert(df_th.equals(df_tbx))

def test_event_file_raw(prepare, testdir):
    log_dir = os.path.join(testdir.tmpdir, 'run')
    dirs = os.listdir(log_dir)
    assert len(dirs) == 1
    event_file = os.path.join(log_dir, dirs[0])
    reader = SummaryReader(event_file, pivot=True)
    # Test raw functions
    # - Test `raw_tags` and `get_raw_tags`
    assert reader.raw_tags == reader.get_raw_tags()
    assert reader.raw_tags['histograms'] == reader.get_raw_tags('histograms')
    assert set(reader.raw_tags['histograms']) == {'dist'}
    # - Test `raw_events` and `get_raw_events`
    assert reader.raw_events == reader.get_raw_events()
    assert reader.raw_events['histograms'] == reader.get_raw_events('histograms')
    assert reader.raw_events['histograms']['dist'] == reader.get_raw_events('histograms', 'dist')
    # - Test raw event count & type
    events: List[HistogramEvent] = reader.get_raw_events('histograms', 'dist')
    assert len(events) == N_EVENTS
    assert type(events[0]) == HistogramEvent
    for i in range(N_EVENTS):
        assert events[i].step == i
        assert events[i].histogram_value.num == N_PARTICLES

def check_others(reader):
    assert len(reader.scalars) == 0
    assert len(reader.tensors) == 0
    assert len(reader.hparams) == 0

def test_log_dir(prepare, testdir):
    log_dir = os.path.join(testdir.tmpdir, 'run')
    # Test default
    reader = SummaryReader(log_dir)
    df = reader.histograms
    assert df.columns.to_list() == ['step', 'tag', 'counts', 'limits']
    assert df['step'].to_list() == [i for i in range(N_EVENTS)]
    assert df['tag'].to_list() == ['dist'] * N_EVENTS
    for i in range(N_EVENTS):
        assert len(df['counts'][i]) + 1 == len(df['limits'][i])
        assert sum(df['counts'][i]) == N_PARTICLES
    check_others(reader)
    # Test pivot
    reader = SummaryReader(log_dir, pivot=True)
    assert len(reader.children) == 1
    df = reader.histograms
    assert df.columns.to_list() == ['step', 'dist/counts', 'dist/limits',]
    assert df['step'].to_list() == [i for i in range(N_EVENTS)]
    for i in range(N_EVENTS):
        assert len(df['dist/counts'][i]) + 1 == len(df['dist/limits'][i])
        assert sum(df['dist/counts'][i]) == N_PARTICLES
    check_others(reader)
    # Test all columns
    reader = SummaryReader(log_dir, extra_columns={
                           'min', 'max', 'num', 'sum', 'sum_squares',
                           'wall_time', 'dir_name', 'file_name'})
    df = reader.histograms
    assert df.columns.to_list() == ['step', 'tag', 'counts', 'limits',
                                    'max', 'min', 'num', 'sum', 'sum_squares',
                                    'wall_time', 'dir_name', 'file_name']
    assert df['step'].to_list() == [i for i in range(N_EVENTS)]
    assert df['tag'].to_list() == ['dist'] * N_EVENTS
    for i in range(N_EVENTS):
        assert len(df['counts'][i]) + 1 == len(df['limits'][i])
        assert sum(df['counts'][i]) == N_PARTICLES
        assert np.isscalar(df['min'][i])
        assert np.isscalar(df['max'][i])
        assert df['num'][i] == N_PARTICLES
        assert np.isscalar(df['sum'][i])
        assert np.isscalar(df['sum_squares'][i])
    check_others(reader)
    # Test pivot & all columns
    reader = SummaryReader(log_dir, pivot=True, extra_columns={
                           'min', 'max', 'num', 'sum', 'sum_squares',
                           'wall_time', 'dir_name', 'file_name'})
    df = reader.histograms
    assert df.columns.to_list() == ['step', 'dist/counts', 'dist/limits',
                                    'dist/max', 'dist/min', 'dist/num', 'dist/sum', 'dist/sum_squares',
                                    'wall_time', 'dir_name', 'file_name']
    assert df['step'].to_list() == [i for i in range(N_EVENTS)]
    for i in range(N_EVENTS):
        assert len(df['dist/counts'][i]) + 1 == len(df['dist/limits'][i])
        assert sum(df['dist/counts'][i]) == N_PARTICLES
        assert np.isscalar(df['dist/min'][i])
        assert np.isscalar(df['dist/max'][i])
        assert df['dist/num'][i] == N_PARTICLES
        assert np.isscalar(df['dist/sum'][i])
        assert np.isscalar(df['dist/sum_squares'][i])
    check_others(reader)

def test_histogram_to_cdf():
    counts = [1, 3]
    limits = [-10, 0, 10]
    x = [-20, -11, -10, -9, -1, 0, 1, 9, 10, 11, 20]
    y = SummaryReader.histogram_to_cdf(counts, limits, x)
    y = list(y * np.sum(counts))
    expected_y = [0., 0., 0., 0.1, 0.9, 1, 1+3*0.1, 1+3*0.9, 4., 4., 4.]
    assert y == expected_y

# TODO: histogram doc -> df[...].apply(tensor_to_histogram_dict)
# TODO: histogram test -> also use apply