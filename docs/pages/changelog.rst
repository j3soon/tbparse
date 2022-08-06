===================================
Change Log
===================================

v0.0.7
===================================

Features:

* Support parsing with reduced feature set without using the ``tensorflow`` package

Fixes:

* Fix the behavior of ``event_types`` for ``SummaryReader`` (Breaking Change)
* Explicitly support Python 3.10 on PyPI
* Fix preview images on PyPI

Changes:

* Add benchmarks and profiling results
* Optimize ``_merge_values`` when no merging is required
* Change row-oriented parsing to column-oriented parsing

v0.0.6
===================================

Features:

* Support parsing ``images`` and ``audio``

Fixes:

* Unify ``histogram`` API (Breaking Change)

Docs:

* Add sample IPython notebook

v0.0.5
===================================

Features:

* Support parsing ``text``

v0.0.4
===================================

Features:

* Support parsing ``hparams``

Fixes:

* Fix empty directory bug

v0.0.3
===================================

Changes:

* Clarify SummaryReader's parameters (Breaking Change)

Fixes:

* Fix PyPI package metadata

v0.0.2
===================================

Fixes:

* Fix PyPI packaging issue

v0.0.1
===================================

Features:

* Support parsing ``scalars``
* Support parsing ``tensors``
* Support parsing ``histograms``