import os

import numpy as np
import pytest
import torch
from tbparse import SummaryReader
from torch.utils.tensorboard import SummaryWriter


@pytest.fixture
def prepare(testdir):
    # Ref: https://pytorch.org/docs/stable/tensorboard.html
    log_dir = os.path.join(testdir.tmpdir, 'run')
    vertices_tensor = torch.as_tensor([
        [1, 1, 1],
        [-1, -1, 1],
        [1, -1, -1],
        [-1, 1, -1],
    ], dtype=torch.float).unsqueeze(0)
    colors_tensor = torch.as_tensor([
        [255, 0, 0],
        [0, 255, 0],
        [0, 0, 255],
        [255, 0, 255],
    ], dtype=torch.int).unsqueeze(0)
    faces_tensor = torch.as_tensor([
        [0, 2, 3],
        [0, 3, 1],
        [0, 1, 2],
        [1, 3, 2],
    ], dtype=torch.int).unsqueeze(0)
    writer = SummaryWriter(log_dir)
    writer.add_mesh('my_mesh', vertices=vertices_tensor, colors=colors_tensor, faces=faces_tensor)
    writer.close()

def test_log_dir(prepare, testdir):
    vertices_tensor = np.array([[
        [1, 1, 1],
        [-1, -1, 1],
        [1, -1, -1],
        [-1, 1, -1],
    ]], dtype=np.float)
    colors_tensor = np.array([[
        [255, 0, 0],
        [0, 255, 0],
        [0, 0, 255],
        [255, 0, 255],
    ]], dtype=np.float)
    faces_tensor = np.array([[
        [0, 2, 3],
        [0, 3, 1],
        [0, 1, 2],
        [1, 3, 2],
    ]], dtype=np.float)
    log_dir = os.path.join(testdir.tmpdir, 'run')
    reader = SummaryReader(log_dir, pivot=True)
    df = reader.tensors
    assert df.columns.tolist() == ['step', 'my_mesh_COLOR', 'my_mesh_FACE', 'my_mesh_VERTEX']
    assert df.loc[0, 'step'] == 0
    assert np.array_equal(df.loc[0, 'my_mesh_COLOR'], colors_tensor)
    assert np.array_equal(df.loc[0, 'my_mesh_FACE'], faces_tensor)
    assert np.array_equal(df.loc[0, 'my_mesh_VERTEX'], vertices_tensor)
    # TODO: from tensorboard.plugins.mesh.plugin_data_pb2 import MeshPluginData
    assert False
