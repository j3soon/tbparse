===================================
Parsing without tbparse
===================================

To the best of our knowledge, there are no official documentation on parsing
Tensorboard event files. However, it can be done with TensorFlow in two ways,
which require inspecting the TensorFlow source code.

If you have some event logs that are difficult to parse with the current
``tbparse``, you can ask us by opening an `issue`_. Alternatively,
you may parse the event logs by the methods described below.

Similar to other pages, we need to first generate a event file. We use PyTorch
event writer to log ``scalars`` for simplicity. Since ``tensors`` and other
type of events may require parsing
`Protocol Buffers <https://developers.google.com/protocol-buffers>`_,
which requires needs more parsing code.

   >>> import os
   >>> import tempfile
   >>> from torch.utils.tensorboard import SummaryWriter
   >>> # Prepare temp dirs for storing event files
   >>> tmpdir = tempfile.TemporaryDirectory()
   >>> log_dir = tmpdir.name
   >>> writer = SummaryWriter(log_dir)
   >>> for i in range(5):
   ...   writer.add_scalar('y=2x', i * 2, i)
   >>> writer.close()
   >>> event_file = os.path.join(log_dir, os.listdir(log_dir)[0])

Event Accumulator
===================================

   >>> from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
   >>> event_acc = EventAccumulator(event_file)
   >>> event_acc.Reload() # doctest: +ELLIPSIS
   <tensorboard.backend.event_processing.event_accumulator.EventAccumulator object at ...>
   >>> print(event_acc.Tags())
   {'images': [], 'audio': [], 'histograms': [], 'scalars': ['y=2x'], 'distributions': [], 'tensors': [], 'graph': False, 'meta_graph': False, 'run_metadata': []}
   >>> for e in event_acc.Scalars('y=2x'):
   ...   print(e.step, e.value)
   0 0.0
   1 2.0
   2 4.0
   3 6.0
   4 8.0

* The source code of `Event Accumulator is on GitHub <https://github.com/tensorflow/tensorboard/blob/master/tensorboard/backend/event_processing/event_accumulator.py>`_.
* A minimum example of `Event Accumulator is on Stack Overflow <https://stackoverflow.com/a/45899735/>`_.
* Since :class:`tbparse.SummaryReader` also uses Event Accumulator, more example and usages can be found in the source code section under :ref:`tbparse_api`.

Summary Iterator
===================================

   >>> import tensorflow as tf
   >>> from tensorflow.python.summary.summary_iterator import summary_iterator
   >>> for e in summary_iterator(event_file):
   ...   for v in e.summary.value:
   ...     if v.tag == 'y=2x':
   ...       print(e.step, v.simple_value)
   0 0.0
   1 2.0
   2 4.0
   3 6.0
   4 8.0

* The source of `Summary Iterator is on GitHub <https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/summary/summary_iterator.py>`_.
* A minimum example of `Summary Iterator is on Stack Overflow <https://stackoverflow.com/a/37359199/>`_.

Experiment Data Access API for tensorboard.dev
======================================================================

The offical `tensorboard data access API <https://www.tensorflow.org/tensorboard/dataframe_api>`_
is only available if you are using `tensorboard.dev <https://tensorboard.dev/>`_,
and only supports parsing scalar logs, which makes it not useful for
general cases. The `source code <https://github.com/tensorflow/tensorboard/blob/master/tensorboard/data/experimental/experiment_from_dev.py>`_
of the data acess API parses everything on the tensorboard.dev server,
so it's not possible to modify it for offline use.

Related Tools
===================================

Some other (unofficial) related tools on GitHub:

* `chingyaoc/Tensorboard2Seaborn <https://github.com/chingyaoc/Tensorboard2Seaborn>`_
* `wookayin/tensorboard-tools <https://github.com/wookayin/tensorboard-tools>`_
* `velikodniy/tbparser <https://github.com/velikodniy/tbparser>`_
* `akimach/tfgraphviz <https://github.com/akimach/tfgraphviz>`_
* `ildoonet/tbreader <https://github.com/ildoonet/tbreader>`_
* `mrahtz/tbplot <https://github.com/mrahtz/tbplot>`_

If you know some related tools not listed here,
you can open an `issue`_ or `pull request`_, and I'll add it to the list.

.. _issue: https://github.com/j3soon/tbparse/issues
.. _pull request: https://github.com/j3soon/tbparse/pulls