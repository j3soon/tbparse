import keyword
import os

import numpy as np
import pytest
import torch
from tbparse import SummaryReader
from torch.utils.tensorboard import SummaryWriter


@pytest.fixture
def prepare(testdir):
    # Ref: https://numpy.org/neps/nep-0019-rng-policy.html#supporting-unit-tests
    np.random.seed(1234)
    # Ref: https://pytorch.org/docs/stable/tensorboard.html
    log_dir = os.path.join(testdir.tmpdir, 'run')

    meta = []
    while len(meta)<100:
        meta = meta+keyword.kwlist # get some strings
    meta = meta[:100]

    for i, v in enumerate(meta):
        meta[i] = v+str(i)

    label_img = torch.rand(100, 3, 10, 32)
    for i in range(100):
        label_img[i]*=i/100.0

    # Hack that fixes add_embedding
    # Ref: https://github.com/pytorch/pytorch/issues/30966#issuecomment-582747929
    import tensorflow as tf
    import tensorboard as tb
    tf_io_gfile_backup = tf.io.gfile
    tf.io.gfile = tb.compat.tensorflow_stub.io.gfile

    writer = SummaryWriter(log_dir)
    writer.add_embedding(torch.randn(100, 5), metadata=meta, label_img=label_img)
    writer.add_embedding(torch.randn(100, 5), label_img=label_img)
    writer.add_embedding(torch.randn(100, 5), metadata=meta)
    writer.close()

    tf.io.gfile = tf_io_gfile_backup

@pytest.mark.skip(reason="add_embedding is not supported yet")
def test_log_dir(prepare, testdir):
    log_dir = os.path.join(testdir.tmpdir, 'run')
    reader = SummaryReader(log_dir, pivot=True)
    df = reader.tensors
    print(df.columns)
    print(df)
    assert False
