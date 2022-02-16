import os
import urllib

import pytest
import torch
from PIL import Image
from tbparse import SummaryReader
from torch.utils.tensorboard import SummaryWriter
from torchvision import models, transforms


@pytest.fixture
def prepare(testdir):
    # Ref: https://pytorch.org/docs/stable/tensorboard.html
    # Ref: https://pytorch.org/vision/stable/models.html
    # Ref: https://pytorch.org/hub/pytorch_vision_resnet/
    log_dir = os.path.join(testdir.tmpdir, 'run')

    model = models.resnet18()
    model.eval()

    # Download an example image from the pytorch website
    url, filename = ("https://github.com/pytorch/hub/raw/master/images/dog.jpg", "dog.jpg")
    try: urllib.URLopener().retrieve(url, filename)
    except: urllib.request.urlretrieve(url, filename)

    input_image = Image.open(filename)
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    input_tensor = preprocess(input_image)
    input_batch = input_tensor.unsqueeze(0) # create a mini-batch as expected by the model

    writer = SummaryWriter(log_dir)
    writer.add_graph(model, input_batch)
    writer.close()

@pytest.mark.skip(reason="add_graph is not supported yet")
def test_log_dir(prepare, testdir):
    log_dir = os.path.join(testdir.tmpdir, 'run')
    reader = SummaryReader(log_dir, pivot=True)
    df = reader.tensors
    print(df.columns)
    print(df)
    assert False
