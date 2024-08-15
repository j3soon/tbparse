.. _tbparse_installation:

===================================
Installation
===================================

.. highlight:: sh

Install from PyPI:

.. code-block:: bash

   pip install tensorflow # optional, only required if you want to parse images and audio
   pip install -U tbparse # requires Python >= 3.7

**Note**: For details on when TensorFlow is required, see :ref:`Installing without TensorFlow <tbparse_installing-without-tensorflow>`.

Install from Source:

.. code-block:: bash

   git clone https://github.com/j3soon/tbparse
   cd tbparse
   pip install tensorflow # optional, only required if you want to parse images and audio
   pip install -e . # requires Python >= 3.7

.. _tbparse_installing-without-tensorflow:

Installing without TensorFlow
===================================

You can install tbparse with reduced feature set if you don't want to install TensorFlow:

.. code-block:: bash

   # Don't install TensorFlow
   pip install -U tbparse # requires Python >= 3.7

Without TensorFlow, tbparse supports parsing
:ref:`scalars <tbparse_parsing-scalars>`,
:ref:`tensors <tbparse_parsing-tensors>`,
:ref:`histograms <tbparse_parsing-histograms>`,
:ref:`hparams <tbparse_parsing-hparams>`, and
:ref:`text <tbparse_parsing-text>`.
but doesn't support parsing
:ref:`images <tbparse_parsing-images>` and
:ref:`audio <tbparse_parsing-audio>`.

tbparse will instruct you to install TensorFlow by raising an error if you try to parse the unsupported event types, such as:

   ModuleNotFoundError: No module named 'tensorflow'. Please install 'tensorflow' or 'tensorflow-cpu'.

In addition, an error may occur if you have installed TensorFlow and TensorBoard and uninstalled TensorFlow afterwards:

   AttributeError: module 'tensorflow' has no attribute 'io'

This error occurs since TensorBoard will depend on TensorFlow if TensorFlow exists in the environment.
See `TensorBoard README <https://github.com/tensorflow/tensorboard#can-i-run-tensorboard-without-a-tensorflow-installation>`_
for more information.

To resolve this issue, create a new virtual environment and install tbparse without installing TensorFlow.
Or you may uninstall all packages related to TensorFlow and TensorBoard, which require much more effort.
