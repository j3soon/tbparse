"""
Provides a `SummaryReader` class that will read all tensorboard events and
summaries in a directory contains multiple event files, or a single event file.
"""

from tbparse.summary_reader import SummaryReader

__all__ = ['SummaryReader', ]
