"""
Provides a `SummaryReader` class that will read all tensorboard events and
summaries in a directory contains multiple event files, or a single event file.
"""

from tbparse.summary_reader import SummaryReader
from tbparse import version as _version

__all__ = ['SummaryReader', '__version__', ]

__version__ = _version.VERSION
