import os
import tempfile
from typing import List

import pytest
import tensorboardX
import tensorflow as tf
from tbparse import SummaryReader
from tensorboard.backend.event_processing.event_accumulator import ScalarEvent
from tensorboard.plugins.hparams import api as hp
from tensorboard.plugins.hparams.plugin_data_pb2 import HParamsPluginData
from torch.utils.tensorboard import SummaryWriter

N_RUNS = 3
N_EVENTS = 5

@pytest.fixture
def prepare(testdir):
    # Ref: https://pytorch.org/docs/stable/tensorboard.html
    log_dir = os.path.join(testdir.tmpdir, 'run')
    for i in range(N_RUNS):
        writer = SummaryWriter(os.path.join(log_dir, f'run{i}'))
        writer.add_hparams({'name': 'test', 'run_id': i}, {'metric': i})
        for j in range(N_EVENTS):
            writer.add_scalar('y=2x+C', j * 2 + i, j)
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
    # Note: tensorboardX automatically escapes special characters.
    # Prepare Log
    log_dir_th = os.path.join(testdir.tmpdir, 'run')
    tmpdir_tbx = tempfile.TemporaryDirectory()
    log_dir_tbx = os.path.join(tmpdir_tbx.name, 'run')
    for i in range(N_RUNS):
        writer = tensorboardX.SummaryWriter(os.path.join(log_dir_tbx, f'run{i}'))
        writer.add_hparams({'name': 'test', 'run_id': i}, {'metric': i})
        for j in range(N_EVENTS):
            writer.add_scalar('y=2x+C', j * 2 + i, j)
        writer.close()
    # (default) Parse & Compare
    df_th = SummaryReader(log_dir_th).scalars
    df_tbx = SummaryReader(log_dir_tbx).scalars
    df_th.replace("y=2x+C", "y_2x_C", inplace=True)
    assert df_th.equals(df_tbx)
    df_th = SummaryReader(log_dir_th).hparams
    df_tbx = SummaryReader(log_dir_tbx).hparams
    assert df_th.equals(df_tbx)
    # (pivot) Parse & Compare
    df_th = SummaryReader(log_dir_th, pivot=True).scalars
    df_tbx = SummaryReader(log_dir_tbx, pivot=True).scalars
    df_th.rename(columns={"y=2x+C": "y_2x_C"}, inplace=True)
    assert df_th.equals(df_tbx)
    df_th = SummaryReader(log_dir_th, pivot=True).hparams
    df_tbx = SummaryReader(log_dir_tbx, pivot=True).hparams
    assert df_th.equals(df_tbx)

def test_tensorflow(prepare, testdir):
    # Prepare Log
    log_dir_th = os.path.join(testdir.tmpdir, 'run')
    tmpdir_tf = tempfile.TemporaryDirectory()
    log_dir_tf = os.path.join(tmpdir_tf.name, 'run')
    for i in range(N_RUNS):
        writer = tf.summary.create_file_writer(os.path.join(log_dir_tf, f'run{i}'))
        writer.set_as_default()
        hp.hparams({'name': 'test', 'run_id': i})
        tf.summary.scalar('metric', i, step=0)
        for j in range(N_EVENTS):
            tf.summary.scalar('y=2x+C', j * 2 + i, j)
        writer.close()
    # (default) Parse & Compare
    df_th = SummaryReader(log_dir_th).scalars
    df_tf = SummaryReader(log_dir_tf).tensors
    assert df_th.equals(df_tf)
    df_th = SummaryReader(log_dir_th).hparams
    df_tf = SummaryReader(log_dir_tf).hparams
    assert df_th.equals(df_tf)
    # (pivot) Parse & Compare
    df_th = SummaryReader(log_dir_th, pivot=True).scalars
    df_tf = SummaryReader(log_dir_tf, pivot=True).tensors
    assert df_th.equals(df_tf)
    df_th = SummaryReader(log_dir_th, pivot=True).hparams
    df_tf = SummaryReader(log_dir_tf, pivot=True).hparams
    assert df_th.equals(df_tf)

def get_tmpdir_info(tmpdir):
    log_dir = os.path.join(tmpdir, 'run')
    run_dir = os.path.join(log_dir, 'run0')
    dirs = os.listdir(run_dir)
    assert len(dirs) == 2
    for d in dirs:
        path = os.path.join(run_dir, d)
        if os.path.isfile(path):
            event_file = path
            event_filename = d
        else:
            hp_dir = path
            hp_dirname = d
    dirs = os.listdir(hp_dir)
    assert len(dirs) == 1
    hp_filename = dirs[0]
    hp_file = os.path.join(hp_dir, hp_filename)
    d = {
        'log_dir': log_dir,
        'run_dir': run_dir,
        'event_file': event_file,
        'event_filename': event_filename,
        'hp_dir': hp_dir,
        'hp_dirname': hp_dirname,
        'hp_file': hp_file,
        'hp_filename': hp_filename,
    }
    return d

def test_event_file_raw(prepare, testdir):
    tmpinfo = get_tmpdir_info(testdir.tmpdir)
    reader = SummaryReader(tmpinfo['hp_file'])
    # Test raw functions for scalars
    # - Test `raw_tags` and `get_raw_tags`
    assert reader.raw_tags == reader.get_raw_tags()
    assert reader.raw_tags['scalars'] == reader.get_raw_tags('scalars')
    assert set(reader.raw_tags['scalars']) == {'metric'}
    # - Test `raw_events` and `get_raw_events`
    assert reader.raw_events == reader.get_raw_events()
    assert reader.raw_events['scalars'] == reader.get_raw_events('scalars')
    assert reader.raw_events['scalars']['metric'] == reader.get_raw_events('scalars', 'metric')
    # - Test raw event count & type
    events: List[ScalarEvent] = reader.get_raw_events('scalars', 'metric')
    assert len(events) == 1
    assert type(events[0]) == ScalarEvent
    assert (events[0].step, events[0].value) == (0, 0.0)

    # Test raw functions for hparams
    # - Test `raw_tags` and `get_raw_tags`
    assert reader.raw_tags == reader.get_raw_tags()
    assert reader.raw_tags['hparams'] == reader.get_raw_tags('hparams')
    assert set(reader.raw_tags['hparams']) == {
        '_hparams_/session_end_info',
        '_hparams_/session_start_info',
        '_hparams_/experiment',
    }
    # - Test `raw_events` and `get_raw_events`
    assert reader.raw_events == reader.get_raw_events()
    assert reader.raw_events['hparams'] == reader.get_raw_events('hparams')
    assert reader.raw_events['hparams']['_hparams_/session_end_info'] == \
        reader.get_raw_events('hparams', '_hparams_/session_end_info')
    assert reader.raw_events['hparams']['_hparams_/session_start_info'] == \
        reader.get_raw_events('hparams', '_hparams_/session_start_info')
    assert reader.raw_events['hparams']['_hparams_/experiment'] == \
        reader.get_raw_events('hparams', '_hparams_/experiment')
    # - Test raw event count & type
    events: List[HParamsPluginData] = reader.get_raw_events('hparams')
    assert len(events) == 3
    exp = HParamsPluginData.FromString(events['_hparams_/experiment'])
    ssi = HParamsPluginData.FromString(events['_hparams_/session_start_info'])
    sei = HParamsPluginData.FromString(events['_hparams_/session_end_info'])
    for event in [exp, ssi, sei]:
        assert type(event) == HParamsPluginData
        assert event.version == 0
    exp = exp.experiment
    ssi = ssi.session_start_info
    sei = sei.session_end_info
    # Check ProtoBufs
    # Ref: https://github.com/tensorflow/tensorboard/blob/master/tensorboard/plugins/hparams/api.proto
    # Ref: https://github.com/tensorflow/tensorboard/blob/master/tensorboard/plugins/hparams/plugin_data.proto
    # check exp
    assert exp.name == ''
    assert exp.description == ''
    assert exp.user == ''
    assert exp.time_created_secs == 0.0
    # - exp.hparam_infos
    assert len(exp.hparam_infos) == 2
    info = exp.hparam_infos[0]
    assert info.name == 'name'
    assert info.display_name == ''
    assert info.description == ''
    assert info.type == 1 # DATA_TYPE_STRING
    assert len(info.domain_discrete.values) == 0
    assert info.domain_interval.min_value == 0.0
    assert info.domain_interval.max_value == 0.0
    info = exp.hparam_infos[1]
    assert info.name == 'run_id'
    assert info.display_name == ''
    assert info.description == ''
    assert info.type == 3 # DATA_TYPE_FLOAT64
    assert len(info.domain_discrete.values) == 0
    assert info.domain_interval.min_value == 0.0
    assert info.domain_interval.max_value == 0.0
    # - exp.metric_infos
    assert len(exp.metric_infos) == 1
    info = exp.metric_infos[0]
    assert info.name.group == ''
    assert info.name.tag == 'metric'
    assert info.display_name == ''
    assert info.description == ''
    assert info.dataset_type == 0 # DATA_TYPE_UNSET
    # check ssi
    assert ssi.model_uri == ''
    assert ssi.monitor_url == ''
    assert ssi.group_name == ''
    assert ssi.start_time_secs == 0.0
    # - ssi.hparams
    # google.protobuf.struct_pb2.Value.ListFields()
    fields = ssi.hparams["run_id"].ListFields()
    assert len(fields) == 1
    assert len(fields[0]) == 2
    assert fields[0][1] == 0.0
    fields = ssi.hparams["name"].ListFields()
    assert len(fields) == 1
    assert len(fields[0]) == 2
    assert fields[0][1] == "test"
    # check sei
    assert sei.status == 1 # STATUS_SUCCESS
    assert sei.end_time_secs == 0.0

def check_others(reader):
    assert len(reader.tensors) == 0
    assert len(reader.histograms) == 0

def test_event_file(prepare, testdir):
    tmpinfo = get_tmpdir_info(testdir.tmpdir)
    # default
    reader = SummaryReader(tmpinfo['hp_file'])
    # - test hparams
    assert reader.hparams.columns.to_list() == ['tag', 'value']
    assert reader.hparams['tag'].to_list() == ['name', 'run_id']
    assert reader.hparams['value'].to_list() == ['test', 0.0]
    # - test metric
    assert reader.scalars['tag'].to_list() == ['metric']
    assert reader.scalars['value'].to_list() == [0.0]
    check_others(reader)
    # pivot
    reader = SummaryReader(tmpinfo['hp_file'], pivot=True)
    assert reader.hparams.columns.to_list() == ['name', 'run_id']
    assert reader.hparams['name'].to_list() == ['test']
    assert reader.hparams['run_id'].to_list() == [0.0]
    check_others(reader)
    # all columns
    reader = SummaryReader(tmpinfo['hp_file'], extra_columns={
        'wall_time', 'dir_name', 'file_name'})
    assert reader.hparams.columns.to_list() == ['tag', 'value', 'dir_name', 'file_name']
    assert reader.hparams['tag'].to_list() == ['name', 'run_id']
    assert reader.hparams['value'].to_list() == ['test', 0.0]
    assert reader.hparams['dir_name'].to_list() == [""] * 2
    assert reader.hparams['file_name'].to_list() == [tmpinfo['hp_filename']] * 2
    check_others(reader)
    # pivot & all columns
    reader = SummaryReader(tmpinfo['hp_file'], pivot=True, extra_columns={
        'wall_time', 'dir_name', 'file_name'})
    assert reader.hparams.columns.to_list() == ['name', 'run_id', 'dir_name', 'file_name']
    assert reader.hparams['name'].to_list() == ['test']
    assert reader.hparams['run_id'].to_list() == [0.0]
    assert reader.hparams['dir_name'].to_list() == [""]
    assert reader.hparams['file_name'].to_list() == [tmpinfo['hp_filename']]
    check_others(reader)

def test_run_dir(prepare, testdir):
    tmpinfo = get_tmpdir_info(testdir.tmpdir)
    # default
    reader = SummaryReader(tmpinfo['run_dir'])
    # - test hparams
    assert reader.hparams.columns.to_list() == ['tag', 'value']
    assert reader.hparams['tag'].to_list() == ['name', 'run_id']
    assert reader.hparams['value'].to_list() == ['test', 0.0]
    # - test metric
    assert reader.scalars['tag'].to_list() == ['metric'] + ['y=2x+C'] * N_EVENTS
    assert reader.scalars['value'].to_list() == [0.0] + [i * 2 for i in range(N_EVENTS)]
    check_others(reader)
    # pivot
    reader = SummaryReader(tmpinfo['run_dir'], pivot=True)
    assert reader.hparams.columns.to_list() == ['name', 'run_id']
    assert reader.hparams['name'].to_list() == ['test']
    assert reader.hparams['run_id'].to_list() == [0.0]
    check_others(reader)
    # all columns
    reader = SummaryReader(tmpinfo['run_dir'], extra_columns={
        'wall_time', 'dir_name', 'file_name'})
    assert reader.hparams.columns.to_list() == ['tag', 'value', 'dir_name', 'file_name']
    assert reader.hparams['tag'].to_list() == ['name', 'run_id']
    assert reader.hparams['value'].to_list() == ['test', 0.0]
    assert reader.hparams['dir_name'].to_list() == [tmpinfo['hp_dirname']] * 2
    assert reader.hparams['file_name'].to_list() == [tmpinfo["hp_filename"]] * 2
    check_others(reader)
    # pivot & all columns
    reader = SummaryReader(tmpinfo['run_dir'], pivot=True, extra_columns={
        'wall_time', 'dir_name', 'file_name'})
    assert reader.hparams.columns.to_list() == ['name', 'run_id', 'dir_name', 'file_name']
    assert reader.hparams['name'].to_list() == ['test']
    assert reader.hparams['run_id'].to_list() == [0.0]
    assert reader.hparams['dir_name'].to_list() == [tmpinfo['hp_dirname']]
    assert reader.hparams['file_name'].to_list() == [tmpinfo["hp_filename"]]
    check_others(reader)

def test_log_dir(prepare, testdir):
    tmpinfo = get_tmpdir_info(testdir.tmpdir)
    # default
    reader = SummaryReader(tmpinfo['log_dir'])
    # - test hparams
    assert reader.hparams.columns.to_list() == ['tag', 'value']
    assert reader.hparams['tag'].to_list() == ['name'] * N_RUNS + ['run_id'] * N_RUNS
    assert reader.hparams['value'].to_list() == ['test'] * N_RUNS + [float(i) for i in range(N_RUNS)]
    # - test metric
    assert reader.scalars['tag'].to_list() == ['metric'] * N_RUNS + ['y=2x+C'] * (N_RUNS * N_EVENTS)
    assert reader.scalars['value'].to_list()[:N_RUNS] == [float(i) for i in range(N_RUNS)]
    check_others(reader)
    # pivot
    reader = SummaryReader(tmpinfo['log_dir'], pivot=True)
    assert reader.hparams.columns.to_list() == ['name', 'run_id']
    assert reader.hparams['name'].to_list() == [['test'] * N_RUNS]
    assert reader.hparams['run_id'].to_list() == [[float(i) for i in range(N_RUNS)]]
    check_others(reader)
    # all columns
    reader = SummaryReader(tmpinfo['log_dir'], extra_columns={
        'wall_time', 'dir_name', 'file_name'})
    assert reader.hparams.columns.to_list() == ['tag', 'value', 'dir_name', 'file_name']
    assert len(reader.hparams) == 2 * N_RUNS
    for i in range(N_RUNS):
        s, e = 2*i, 2*(i+1)
        assert reader.hparams['tag'].to_list()[s:e] == ['name', 'run_id']
        assert reader.hparams['value'].to_list()[s:e] == ['test', float(i)]
        assert reader.hparams['dir_name'].to_list()[s].startswith(f"run{i}/")
        assert reader.hparams['dir_name'].to_list()[s] == reader.hparams['dir_name'].to_list()[s+1]
        assert reader.hparams['file_name'].to_list()[s] == reader.hparams['file_name'].to_list()[s+1]
    check_others(reader)
    # pivot & all columns
    reader = SummaryReader(tmpinfo['log_dir'], pivot=True, extra_columns={
        'wall_time', 'dir_name', 'file_name'})
    assert reader.hparams.columns.to_list() == ['name', 'run_id', 'dir_name', 'file_name']
    assert reader.hparams['name'].to_list() == ['test'] * N_RUNS
    assert reader.hparams['run_id'].to_list() == [float(i) for i in range(N_RUNS)]
    assert len(reader.hparams['dir_name']) == N_RUNS
    assert len(reader.hparams['file_name']) == N_RUNS
    for i in range(N_RUNS):
        assert reader.hparams['dir_name'].to_list()[i].startswith(f"run{i}/")
    check_others(reader)
