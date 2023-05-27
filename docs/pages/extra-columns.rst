.. _tbparse_extra-columns:

===================================
Extra Columns
===================================

This page list out the available options for the `extra_columns` argument.

.. contents:: Table of Contents
    :depth: 2
    :local:

All Events
===================================

Please refer to `test_scalar.py <https://github.com/j3soon/tbparse/blob/master/tests/test_summary_reader/test_scalar.py>`_ or
`test_tensor.py <https://github.com/j3soon/tbparse/blob/master/tests/test_summary_reader/test_tensor.py>`_.

`dir_name`
-----------------------------------


Please refer to the :ref:`tbparse_load-log-directory` page.

`file_name`
-----------------------------------

Please refer to the python tests above.

`wall_time`
-----------------------------------

Please refer to the python tests above.

Histogram Events
===================================

`min`, `max`, `num`, `sum`, `sum_squares`
-----------------------------------

Please refer to `test_histogram.py <https://github.com/j3soon/tbparse/blob/master/tests/test_summary_reader/test_histogram.py>`_.

Image Events
===================================

`width`, `height`
-----------------------------------

Please refer to `test_image.py <https://github.com/j3soon/tbparse/blob/master/tests/test_summary_reader/test_image.py>`_.

Audio Events
===================================

`content_type`, `length_frames`, `sample_rate`
-----------------------------------

Please refer to `test_audio.py <https://github.com/j3soon/tbparse/blob/master/tests/test_summary_reader/test_audio.py>`_.
