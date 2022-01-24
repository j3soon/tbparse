import os

import pandas
import pytest
from tbparse import SummaryReader
from torch.utils.tensorboard import SummaryWriter

N_RUNS = 3
N_EVENTS = 5

@pytest.fixture
def prepare(testdir):
    # Ref: https://pytorch.org/docs/stable/tensorboard.html
    log_dir = os.path.join(testdir.tmpdir, 'run')
    for i in range(N_RUNS):
        run_dir = os.path.join(log_dir, f'run{i}')
        writer = SummaryWriter(run_dir)
        # writer.add_hparams({'name': 'test', 'run id': i}, {}, run_name='.')
        for j in range(N_EVENTS):
            writer.add_scalar('y=2x+C', j * 2 + i, j)
            writer.add_scalar('y=3x+C', j * 3 + i, j)
        # non event file
        with open(os.path.join(run_dir, 'temp.txt'), 'w') as file:
            file.write('temp')
        writer.close()
    temp_dir = os.path.join(testdir.tmpdir, 'temp')
    os.mkdir(temp_dir)
    with open(os.path.join(temp_dir, 'temp.txt'), 'w') as file:
        file.write('temp')
    """
    run
    ├── run0
    │   └── events.out.tfevents.<id-1>
    ├── run1
    │   └── events.out.tfevents.<id-2>
    └── run2
        └── events.out.tfevents.<id-3>
    """

def test_empty_dir(prepare, testdir):
    temp_dir = os.path.join(testdir.tmpdir, 'temp')
    reader = SummaryReader(temp_dir)
    assert reader.scalars.columns.to_list() == []

def test_event_file(prepare, testdir):
    log_dir = os.path.join(testdir.tmpdir, 'run')
    run_dir = os.path.join(log_dir, 'run0')
    dirs = sorted(os.listdir(run_dir))
    assert len(dirs) == 2
    event_filename = dirs[0]
    # Test pivot
    reader = SummaryReader(run_dir, pivot=True, extra_columns={
                           'wall_time', 'dir_name', 'file_name'})
    assert len(reader.children) == 2
    assert reader.scalars.columns.to_list(
    ) == ['step', 'y=2x+C', 'y=3x+C', 'wall_time', 'dir_name', 'file_name']
    assert reader.scalars['step'].to_list() == [i for i in range(N_EVENTS)]
    assert reader.scalars['y=2x+C'].to_list() == [i *
                                                  2 for i in range(N_EVENTS)]
    assert reader.scalars['y=3x+C'].to_list() == [i *
                                                  3 for i in range(N_EVENTS)]
    assert len(reader.scalars['wall_time']) == N_EVENTS
    assert len(reader.scalars['wall_time'][0]) == 2
    assert reader.scalars['dir_name'].to_list() == [''] * N_EVENTS
    assert reader.scalars['file_name'].to_list(
    ) == [event_filename] * N_EVENTS

def test_event_types(prepare, testdir):
    log_dir = os.path.join(testdir.tmpdir, 'run')
    run_dir = os.path.join(log_dir, 'run0')
    dirs = sorted(os.listdir(run_dir))
    assert len(dirs) == 2
    event_filename = dirs[0]
    event_file = os.path.join(run_dir, event_filename)
    # Test default
    reader = SummaryReader(event_file, event_types={'tensors'})
    assert reader.scalars.columns.to_list() == []

def test_get_tags(prepare, testdir):
    log_dir = os.path.join(testdir.tmpdir, 'run')
    run_dir = os.path.join(log_dir, 'run0')
    dirs = sorted(os.listdir(run_dir))
    assert len(dirs) == 2
    event_filename = dirs[0]
    event_file = os.path.join(run_dir, event_filename)
    # Test default
    reader = SummaryReader(event_file)
    assert reader.tags['scalars'] == ['y=2x+C', 'y=3x+C']
    assert reader.get_tags('scalars') == ['y=2x+C', 'y=3x+C']
    reader = SummaryReader(run_dir)
    assert reader.tags['scalars'] == ['y=2x+C', 'y=3x+C']
    assert reader.get_tags('scalars') == ['y=2x+C', 'y=3x+C']

# TODO: tags duplicate with file_name, dir_name, etc.
# TODO: log single letter?
# TODO: order difference when pd.concat