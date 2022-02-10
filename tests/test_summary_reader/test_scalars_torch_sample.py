import os
import tempfile

import numpy as np
import pytest
import tensorboardX
from numpy.testing import assert_almost_equal
from tbparse import SummaryReader
from torch.utils.tensorboard import SummaryWriter

R = 5
N_STEPS = 100

@pytest.fixture
def prepare(testdir):
    # Use torch for main tests, logs for tensorboard and tensorboardX are
    # generated in their own tests.
    # Ref: https://pytorch.org/docs/stable/tensorboard.html
    log_dir = os.path.join(testdir.tmpdir, 'run')
    writer = SummaryWriter(log_dir)
    for i in range(N_STEPS):
        writer.add_scalars('run_14h', {'xsinx':i*np.sin(i/R),
                                        'xcosx':i*np.cos(i/R),
                                        'tanx': np.tan(i/R)}, i)
    writer.close()
    """
    run
    ├── events.out.tfevents.<id-1>
    ├── run_14h_tanx
    │   └── events.out.tfevents.<id-2>
    ├── run_14h_xcosx
    │   └── events.out.tfevents.<id-3>
    └── run_14h_xsinx
        └── events.out.tfevents.<id-4>
    """

def test_tensorflow(prepare, testdir):
    pass
    # Note: tensorflow does not allow users to log multiple scalars.

def test_tensorboardX(prepare, testdir):
    # Note: tensorboardX uses '/' instead of '_' for adding scalars.
    # Prepare Log
    log_dir_th = os.path.join(testdir.tmpdir, 'run')
    tmpdir_tbx = tempfile.TemporaryDirectory()
    log_dir_tbx = os.path.join(tmpdir_tbx.name, 'run')
    writer = tensorboardX.SummaryWriter(log_dir_tbx)
    for i in range(N_STEPS):
        writer.add_scalars('run_14h', {'xsinx':i*np.sin(i/R),
                                        'xcosx':i*np.cos(i/R),
                                        'tanx': np.tan(i/R)}, i)
    writer.close()
    # (default) Parse & Compare
    df_th = SummaryReader(log_dir_th).scalars
    df_tbx = SummaryReader(log_dir_tbx).scalars
    assert(df_th.equals(df_tbx))
    # (pivot) Parse & Compare
    df_th = SummaryReader(log_dir_th, pivot=True).scalars
    df_tbx = SummaryReader(log_dir_tbx, pivot=True).scalars
    assert(df_th.equals(df_tbx))
    # (dir_name) Parse & Compare
    df_th = SummaryReader(log_dir_th, extra_columns={'dir_name'}).scalars
    df_tbx = SummaryReader(log_dir_tbx, extra_columns={'dir_name'}).scalars
    for i in range(len(df_tbx)):
        replaced = list(df_th['dir_name'][i])
        replaced[len('run_14h')] = '/'
        replaced = ''.join(replaced)
        assert replaced == df_tbx['dir_name'][i]
    df_th.drop(columns=['dir_name'], inplace=True)
    df_tbx.drop(columns=['dir_name'], inplace=True)
    assert(df_th.equals(df_tbx))
    # (pivot & dir_name) Parse & Compare
    df_th = SummaryReader(log_dir_th, pivot=True, extra_columns={'dir_name'}).scalars
    df_tbx = SummaryReader(log_dir_tbx, pivot=True, extra_columns={'dir_name'}).scalars
    for i in range(len(df_tbx)):
        replaced = list(df_th['dir_name'][i])
        replaced[len('run_14h')] = '/'
        replaced = ''.join(replaced)
        assert replaced == df_tbx['dir_name'][i]
    df_th.drop(columns=['dir_name'], inplace=True)
    df_tbx.drop(columns=['dir_name'], inplace=True)
    assert(df_th.equals(df_tbx))

def test_log_dir(prepare, testdir):
    log_dir = os.path.join(testdir.tmpdir, 'run')
    # Test pivot
    reader = SummaryReader(log_dir, pivot=True, extra_columns={'dir_name'})
    assert len(reader.children) == 4
    assert reader.scalars.columns.to_list() == ['step', 'run_14h', 'dir_name']
    df0 = reader.scalars
    assert df0.shape == (N_STEPS*3, 3)
    steps = [i for i in range(N_STEPS)]
    # xsinx
    df = df0.loc[df0['dir_name'] == 'run_14h_xsinx', ['step', 'run_14h']]
    assert df.shape == (100, 2)
    assert df['step'].to_list() == steps
    assert_almost_equal(df['run_14h'].to_numpy(), [i*np.sin(i/R) for i in range(100)], 2)
    # xcosx
    df = df0.loc[df0['dir_name'] == 'run_14h_xcosx', ['step', 'run_14h']]
    assert df.shape == (100, 2)
    assert df['step'].to_list() == steps
    assert_almost_equal(df['run_14h'].to_numpy(), [i*np.cos(i/R) for i in range(100)], 2)
    # tanx
    df = df0.loc[df0['dir_name'] == 'run_14h_tanx', ['step', 'run_14h']]
    assert df.shape == (100, 2)
    assert df['step'].to_list() == steps
    assert_almost_equal(df['run_14h'].to_numpy(), [np.tan(i/R) for i in range(100)], 2)
    # Test all columns
    reader = SummaryReader(log_dir, extra_columns={
                           'wall_time', 'dir_name', 'file_name'})
    assert reader.scalars.columns.to_list() == ['step', 'tag', 'value', 'wall_time', 'dir_name', 'file_name']
