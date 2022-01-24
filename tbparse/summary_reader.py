"""
Provides a `SummaryReader` class that will read all tensorboard events and
summaries in a directory contains multiple event files, or a single event file.
"""

import copy
import os
from typing import Any, Dict, List, Optional, Set, Tuple, Union, cast

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorboard.backend.event_processing.event_accumulator import (
    AUDIO, COMPRESSED_HISTOGRAMS, HISTOGRAMS, IMAGES, SCALARS,
    STORE_EVERYTHING_SIZE_GUIDANCE, TENSORS, EventAccumulator, HistogramEvent,
    ScalarEvent, TensorEvent)

# from tensorboard.backend.event_processing.event_accumulator import \
#     EventAccumulator, IMAGES, AUDIO, HISTOGRAMS, SCALARS, \
#     COMPRESSED_HISTOGRAMS, TENSORS, GRAPH, META_GRAPH, RUN_METADATA, \
#     STORE_EVERYTHING_SIZE_GUIDANCE, HistogramEvent, ScalarEvent, TensorEvent

# HPARAMS = 'hparams'

MINIMUM_SIZE_GUIDANCE = {
    COMPRESSED_HISTOGRAMS: 1,
    IMAGES: 1,
    AUDIO: 1,
    SCALARS: 1,
    HISTOGRAMS: 1,
    TENSORS: 1,
}

ALL_EVENT_TYPES = {SCALARS, TENSORS, HISTOGRAMS}
ALL_EXTRA_COLUMNS = {'dir_name', 'file_name', 'wall_time', 'min', 'max',
                     'num', 'sum', 'sum_squares'}


class SummaryReader():
    """
    Creates a `SummaryReader` that reads all tensorboard events and summaries
    stored in a event file or a directory containing multiple event files.
    """

    def __init__(self, log_path: str, *, pivot=False, extra_columns=None,
                 event_types=None):
        """The constructor of SummaryReader. Columns contains `step`, `tag`, \
           and `value` by default.

        :param log_path: Load directory location, or load file location.
        :type log_path: str

        :param pivot: Returns long format DataFrame by default, \
                returns wide format DataFrame if set to True. If there are \
                multiple values per step with the same tag, the values are \
                merged into a list.
        :type pivot: bool

        :param extra_columns: Specifies extra columns, defaults to `None`.
                - dir_name:  add a column that contains the relative \
                             directory path.
                - file_name: add a column that contains the relative \
                             event file path.
                - wall_time: add a column that stores the event timestamp.
                - min (histogram): the min value.
                - max (histogram): the max value.
                - num (histogram): the number of values.
                - sum (histogram): the sum of all values.
                - sum_squares (histogram): the sum of squares for all values.
        :type extra_columns: Set[{'dir_name', 'file_name', 'wall_time', \
                'min', 'max', 'num', 'sum', 'sum_squares'}]

        :param event_types: Specifies the event types to parse, \
            defaults to all event types.
        :type event_types: Set[{'scalars', 'tensors', 'histograms'}]
        """
        self._log_path: str = log_path
        """Load directory location, or load file location."""
        self._extra_columns: Set[str] = (extra_columns or set()).copy()
        """Specifies additional required columns."""
        if not isinstance(self._extra_columns, set):
            raise ValueError(f"`columns` should be a {set} instead of \
                              {str(type(self._extra_columns))}")
        diff = self._extra_columns - ALL_EXTRA_COLUMNS
        if len(diff) > 0:
            raise KeyError(f"Invalid columns entries: {diff}")
        self._pivot: bool = pivot
        """Determines whether the DataFrame is stored in wide format."""
        self._event_types: Set[str] = (event_types or ALL_EVENT_TYPES).copy()
        """Specifies the event types to parse."""
        if not isinstance(self._event_types, set):
            raise ValueError(f"`event_types` should be a {set} instead of \
                              {str(type(self._event_types))}")
        diff = self._event_types - ALL_EVENT_TYPES
        if len(diff) > 0:
            raise KeyError(f"Invalid event types: {diff}")
        self._children: Dict[str, 'SummaryReader'] = {}
        """Holds a list of references to the `SummaryReader` children."""

        self._tags: Optional[Dict[str, List[str]]] = None
        """Stores a dictionary contatining a list of parsed tag names for each
        event type."""
        self._events: Dict[str, pd.DataFrame] = self._make_empty_dict(None)
        """Stores a `pandas.DataFrame` containing all events."""

        if not os.path.exists(self.log_path):
            raise ValueError(f"File or directory not found: {self.log_path}")
        if os.path.isfile(self.log_path):
            # Note: tensorflow.python.summary.summary_iterator is less
            #       straightforward, so we use EventAccumulator instead.
            size_guidance = MINIMUM_SIZE_GUIDANCE.copy()
            for e in self._event_types:
                size_guidance[e] = 0  # store everything
            event_acc = EventAccumulator(self.log_path, size_guidance)
            event_acc.Reload()
            self._tags = self._make_empty_dict([])
            for e in self._event_types:
                self._parse_events(e, event_acc=event_acc)
        else:
            # Populate children
            for filename in sorted(os.listdir(self.log_path)):
                filepath = os.path.join(self.log_path, filename)
                r = SummaryReader(filepath,
                                  pivot=self._pivot,
                                  extra_columns=self._extra_columns)
                self._children[filename] = r

    @property
    def log_path(self) -> str:
        """Load directory location, or load file location.

        :return: A directory path or file path.
        :rtype: str
        """
        return self._log_path

    @property
    def tags(self) -> Dict[str, List[str]]:
        """Returns a dictionary contatining a list of parsed tag names for each
        event type.

        :return: A `{eventType: ['list', 'of', 'tags']}` dictionary.
        :rtype: Dict[str, List[str]]
        """
        return cast(Dict[str, List[str]], self.get_tags())

    def get_tags(self, event_type: str = None) -> \
            Union[List[str], Dict[str, List[str]]]:
        """Returns a list of tag names for the specified event type. If
        `event_type` is None, return a dictionary containing a list of tag
        names for each event type.

        :param event_type: the event type to retrieve, None means return all, \
        defaults to None.
        :type event_type: {None, 'histograms', 'scalars', 'tensors'}, optional
        :raises ValueError: if `event_type` is unknown.
        :return: A `['list', 'of', 'tags']` list, or a \
            `{eventType: ['list', 'of', 'tags']}` dictionary.
        :rtype: List[str] | Dict[str, List[str]]
        """
        if event_type not in {None, 'histograms', 'scalars', 'tensors'}:
            raise ValueError(f"Unknown event_type: {event_type}")
        if self._tags is not None:
            # Leaf node returns directly
            if event_type is not None:
                return self._tags[event_type].copy()
            return copy.deepcopy(self._tags)
        # Non-leaf node collects children's tags then return
        tags = self._make_empty_dict([])
        if event_type is not None:
            # Only collect the specified event type
            tags = {event_type: tags[event_type]}
        for t in tags:
            for c in self.children.values():
                # Collect children's tags
                tags[t] += c.get_tags(t)
            # Deduplicate same tag names
            tags[t] = list(dict.fromkeys(tags[t]))
        if event_type is not None:
            return tags[event_type]
        return tags

    @staticmethod
    def _merge_values(s: pd.Series):
        """Merge multiple values. Ignore NaNs, concat others."""
        # Note:
        # Does not support python3.6 since DataFrame does not fully support
        # `np.ndarray` as an element in cell. See the following:
        # lib/python3.6/site-packages/pandas/core/groupby/generic.py:482
        # Python 3.6 EOL: 2021-12-23 (https://www.python.org/downloads/)
        assert isinstance(s, pd.Series)
        x: List[Any] = list(s)
        lst = []
        for xx in x:
            if isinstance(xx, list):
                lst.extend(xx)
            elif np.isscalar(xx):
                if not np.isnan(xx):
                    lst.append(xx)
            else:
                lst.append(xx)
        if len(lst) == 0:
            return np.nan
        if len(lst) == 1:
            return lst[0]
        return lst

    def get_events(self, event_type: str) -> pd.DataFrame:
        """Construct a `pandas.DataFrame` that stores all `event_type` events \
        under `log_path`. Some processing is performed when evaluating this \
        property. Therefore you may want to store the results and reuse it \
        for better performance.

        :type event_type: {None, 'histograms', 'scalars', 'tensors'}.
        :raises ValueError: if `event_type` is unknown.
        :return: A `DataFrame` storing all `event_type` events.
        :rtype: pandas.DataFrame
        """
        if event_type not in ALL_EVENT_TYPES:
            raise ValueError(f"Unknown event_type: {event_type}")
        group_columns = []
        for c in ['dir_name', 'file_name']:
            if c in self._extra_columns:
                group_columns.append(c)
        group_columns.append('step')

        dfs = []
        if os.path.isfile(self.log_path):
            # Leaf node appends events directly
            dfs.append(self._events[event_type])
        else:
            # Non-leaf node collects children's events
            for child in self._children.values():
                df = child.get_events(event_type)
                # iteratively prepend dir_name
                if 'dir_name' in self._extra_columns and \
                        os.path.isdir(child.log_path):
                    dir_name = os.path.basename(child.log_path)
                    df_cond = (df['dir_name'] == '')
                    df.loc[df_cond, 'dir_name'] = dir_name
                    df.loc[~df_cond, 'dir_name'] = \
                        dir_name + '/' + df.loc[~df_cond, 'dir_name']
                dfs.append(df)
        dfs = list(filter(lambda x: x is not None, dfs))
        if len(dfs) == 0:
            return pd.DataFrame()
        df_stacked = pd.concat(dfs, ignore_index=True)
        if df_stacked.empty:
            return pd.DataFrame()
        if not self._pivot:
            group_columns = group_columns[:-1]
            group_columns.extend(['tag', 'step'])
            df_stacked.sort_values(group_columns, ignore_index=True,
                                   inplace=True)
            return df_stacked  # keep original order since no merging occurs

        df_stacked.sort_values(group_columns, ignore_index=True, inplace=True)
        # Merge if there are multiple values per step with the same tag
        grouped = df_stacked.groupby(group_columns, sort=False)
        df = grouped.agg(self._merge_values)
        df.reset_index(inplace=True)
        # Reorder columns
        columns = [x for x in df.columns if x not in
                   ['step', 'wall_time', 'dir_name', 'file_name']]
        columns = ['step'] + columns
        for c in ['wall_time', 'dir_name', 'file_name']:
            if c in self._extra_columns:
                columns.append(c)
        return df[columns]  # reorder

    @property
    def scalars(self) -> pd.DataFrame:
        """Construct a `pandas.DataFrame` that stores all scalar events under \
        `log_path`. Some processing is performed when evaluating this \
        property. Therefore you may want to store the results and reuse it \
        for better performance.

        :return: A `DataFrame` storing all scalar events.
        :rtype: pandas.DataFrame
        """
        return self.get_events(SCALARS)

    @property
    def tensors(self) -> pd.DataFrame:
        """Construct a `pandas.DataFrame` that stores all tensor events under \
        `log_path`. Some processing is performed when evaluating this \
        property. Therefore you may want to store the results and reuse it \
        for better performance.

        :return: A `DataFrame` storing all tensor events.
        :rtype: pandas.DataFrame
        """
        return self.get_events(TENSORS)

    @property
    def histograms(self) -> pd.DataFrame:
        """Construct a `pandas.DataFrame` that stores all histograms events
        under `log_path`. Some processing is performed when evaluating this \
        property. Therefore you may want to store the results and reuse it \
        for better performance.

        :return: A `DataFrame` storing all histograms events.
        :rtype: pandas.DataFrame
        """
        return self.get_events(HISTOGRAMS)

    @staticmethod
    def buckets_to_histogram_dict(lst: List[List[float]]) -> Dict[str, Any]:
        """Convert a list of buckets to histogram dictionary.

        :param lst: A `[['bucket lower', 'bucket upper', 'bucket count']]` \
        list. The range of the bucket is [lower, upper)
        :type lst: List[List[float]]
        :return: A `{hist_data_name: hist_data}` dictionary.
        :rtype: Dict[str, Any]
        """
        limits = []
        counts = []
        for e in lst:
            limits.append(e[0])
            counts.append(e[2])
        limits.append(lst[-1][1])
        assert len(limits) == len(counts) + 1
        d = {
            'limits': np.array(limits),
            'counts': np.array(counts),
            'min': limits[0],
            'max': limits[-1],
            'num': np.sum(counts),
            'sum': np.nan,
            'sum_squares': np.nan,
        }
        return d

    @staticmethod
    def histogram_to_pdf(counts: np.ndarray, limits: np.ndarray,
                         x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Given an array of `x` (values), returns the pair (`c`, `y`), which
        are the corresponding `c` (bucket center) and the linear interpolation
        of its `y` (probability density in bucket), given the bucket counts
        and limits.

        :param counts: The number of values inside the buckets.
        :type counts: np.ndarray
        :param limits: The edges of the buckets.
        :type limits: np.ndarray
        :param x: The input values of x.
        :type x: np.ndarray
        :return: The tuple containing the bucket center and the \
            probability density of the bucket.
        :rtype: Tuple[np.ndarray, np.ndarray]
        """
        y = SummaryReader.histogram_to_cdf(counts, limits, x)
        center = (x[1:]+x[:-1])/2
        density = (y[1:]-y[:-1])/(x[1:]-x[:-1])
        return center, density

    @staticmethod
    def histogram_to_cdf(counts: np.ndarray, limits: np.ndarray,
                         x: np.ndarray) -> np.ndarray:
        """Given an array of `x` (values), returns the linear interpolation of
        its corresponding `y` (cumulative probability), given the bucket counts
        and limits.

        :param counts: The number of values inside the buckets.
        :type counts: np.ndarray
        :param limits: The edges of the buckets.
        :type limits: np.ndarray
        :param x: The input values of x coordinates.
        :type x: np.ndarray
        :return: `y`, the cumulative probability at values `x`.
        :rtype: np.ndarray
        """
        assert len(counts) + 1 == len(limits)
        counts = np.array(counts)
        limits = np.array(limits)
        n = np.sum(counts)
        x = np.array(x)
        # x must be increasing
        assert np.all(np.diff(x) > 0)

        cumsum = np.cumsum(counts)
        cumsum = np.insert(cumsum, 0, [0])
        assert len(cumsum) == len(limits)

        y: List[int] = []
        # Calculate y[i], where x[i] <= limits[0]
        i = 0
        while i < len(x) and x[i] <= limits[0]:
            y.append(0)
            i += 1
        # Calculate y[i], where limits[0] < x[i] <= limits[-1]
        idx = 0
        while i < len(x) and idx + 1 < len(limits):
            if limits[idx+1] < x[i]:
                idx += 1
                continue
            lower = limits[idx]
            upper = limits[idx+1]
            assert lower < x[i] and x[i] <= upper
            assert (x[i] - lower) > 0
            interp = (cumsum[idx] * (upper - x[i]) +
                      cumsum[idx+1] * (x[i] - lower))
            interp /= (upper - lower)
            y.append(interp)
            i += 1
        # Calculate y[i], where limits[-1] < x[i]
        while i < len(x):
            y.append(n)
            i += 1
        return np.array(y) / n

    def _add_columns_scalar(self, d: Dict[str, Any], tag: str, e: ScalarEvent):
        """Add entries in dictionary `d` based on the ScalarEvent `e`"""
        if self._pivot:
            d[tag] = e.value
        else:
            d['tag'] = tag
            d['value'] = e.value

    def _add_columns_tensor(self, d: Dict[str, Any], tag: str, e: TensorEvent):
        """Add entries in dictionary `d` based on the TensorEvent `e`"""
        value = tf.make_ndarray(e.tensor_proto)
        if value.shape == ():
            value = value.item()
        if self._pivot:
            d[tag] = value
        else:
            d['tag'] = tag
            d['value'] = value

    def _add_columns_histograms(self, d: Dict[str, Any], tag: str,
                                e: HistogramEvent):
        """Add entries in dictionary `d` based on the HistogramEvent `e`"""
        hv = e.histogram_value
        limits = np.array([hv.min] + hv.bucket_limit, dtype=np.float64)
        counts = np.array(hv.bucket, dtype=np.float64)
        columns = {
            'limits': limits,
            'counts': counts,
            'min': hv.min,
            'max': hv.max,
            'num': hv.num,
            'sum': hv.sum,
            'sum_squares': hv.sum_squares,
        }
        if not self._pivot:
            d['tag'] = tag
        lst = list(self._extra_columns) + ['limits', 'counts']
        for k, v in columns.items():
            if k in lst:
                key = k if not self._pivot else tag + '/' + k
                d[key] = v

    def _parse_events(self, event_type: str, event_acc: EventAccumulator):
        """Parse and store `event_type` events inside a event file.

        :param event_acc: A loaded `EventAccumulator` for parsing events.
        :type event_acc: EventAccumulator
        :raises ValueError: if `log_path` is a directory.
        """
        if os.path.isdir(self.log_path):
            raise ValueError(f"Not an event file: {self.log_path}")
        rows = []
        assert self._tags is not None
        self._tags[event_type] = event_acc.Tags()[event_type]
        # Add columns that depend on event types
        if event_type == SCALARS:
            add_columns = self._add_columns_scalar
            getter = event_acc.Scalars
        elif event_type == TENSORS:
            add_columns = self._add_columns_tensor
            getter = event_acc.Tensors
        elif event_type == HISTOGRAMS:
            add_columns = self._add_columns_histograms
            getter = event_acc.Histograms
        else:
            raise ValueError(f"Unknown event_type: {event_type}")
        # Add shared columns
        for tag in self._tags[event_type]:
            events = getter(tag)
            for e in events:
                d = {'step': e.step}
                add_columns(d, tag, e)
                if 'wall_time' in self._extra_columns:
                    d['wall_time'] = e.wall_time
                if 'dir_name' in self._extra_columns:
                    d['dir_name'] = ''
                if 'file_name' in self._extra_columns:
                    d['file_name'] = os.path.basename(self.log_path)
                rows.append(d)
        df = pd.DataFrame(rows)
        self._events[event_type] = df

    @property
    def children(self) -> Dict[str, 'SummaryReader']:
        """Returns a list of references to the children `SummaryReader` s.
        Since each child may have their own children, the underlying data
        structure is actually a tree that mirrors the directories and files in
        the file system.

        :return: A `{childName: SummaryReader}` dictionary.
        :rtype: Dict[str, 'SummaryReader']
        """
        return self._children.copy()

    @property
    def raw_tags(self) -> Dict[str, List[str]]:
        """Returns a dictionary containing a list of raw tags for each raw event type.
        This property is only supported when `log_path` is a event file.

        :return: A `{eventType: ['list', 'of', 'tags']}` dictionary.
        :rtype: Dict[str, List[str]]
        """
        return cast(Dict[str, List[str]], self.get_raw_tags())

    def get_raw_tags(self, event_type: str = None) -> \
            Union[List[str], Dict[str, List[str]]]:
        """Returns a list of raw tags for the specified raw event type. If
        `event_type` is None, return a dictionary containing a list of raw
        tags for each raw event type. This function is only supported when
        `log_path` is a event file.

        :param event_type: the event type to retrieve, None means return all, \
            defaults to None.
        :type event_type: {None, 'images', 'audio', 'histograms', 'scalars', \
            'tensors', 'graph', 'meta_graph', 'run_metadata' \
            }, optional
        :raises ValueError: if `log_path` is a directory.
        :raises ValueError: if `event_type` is unknown.
        :return: A `['list', 'of', 'tags']` list, or a \
            `{eventType: ['list', 'of', 'tags']}` dictionary.
        :rtype: List[str] | Dict[str, List[str]]
        """
        if event_type not in {None, 'images', 'audio', 'histograms', 'scalars',
                              'tensors', 'graph', 'meta_graph',
                              'run_metadata'}:
            raise ValueError(f"Unknown event_type: {event_type}")
        if os.path.isdir(self.log_path):
            raise ValueError(f"Not an event file: {self.log_path}")
        event_acc = EventAccumulator(
            self.log_path, STORE_EVERYTHING_SIZE_GUIDANCE)
        event_acc.Reload()
        if event_type is None:
            return event_acc.Tags()
        return event_acc.Tags()[event_type]

    @property
    def raw_events(self) -> Dict[str, Dict[str, List[Any]]]:
        """Returns a dictionary of dictionary containing a list of
        raw events for each raw event type.
        This property is only supported when `log_path` is a event file.

        :return: A `{eventType: {tag: ['list', 'of', 'events']}}` dictionary.
        :rtype: Dict[str, Dict[str, List[Any]]]
        """
        return cast(Dict[str, Dict[str, List[Any]]], self.get_raw_events())

    def get_raw_events(self, event_type: str = None, tag: str = None) \
            -> Union[List[Any], Dict[str, List[Any]],
                     Dict[str, Dict[str, List[Any]]]]:
        """Returns a list of raw events for the specified raw event type. If
        `tag` is None, return a dictionary containing a list of raw events for
        each raw event type. If `event_type` is None, return a dictionary of
        dictionary containing a list of raw events for each raw event type.
        This function is only supported when `log_path` is a event file.

        :raises ValueError: if `log_path` is a directory.
        :raises KeyError: if `event_type` is unknown.
        :raises KeyError: If the `tag` is not found.
        :return: A `['list', 'of', 'events']` list, or a \
            `{tag: ['list', 'of', 'events']}` dictionary, or a \
            `{eventType: {tag: ['list', 'of', 'events']}}` dictionary.
        :rtype: List[Any] | Dict[str, List[Any]] | \
                Dict[str, Dict[str, List[Any]]]
        """
        if os.path.isdir(self.log_path):
            raise ValueError(f"Not an event file: {self.log_path}")
        event_acc = EventAccumulator(
            self.log_path, STORE_EVERYTHING_SIZE_GUIDANCE)
        event_acc.Reload()
        if event_type is None:
            # Return all event types by recursion
            if tag is not None:
                raise ValueError("tag shouldn't be set if event_type is None")
            lst = self._make_empty_dict([])
            for t in lst:
                # Collect children's events
                events = self.get_raw_events(t)
                lst[t] = cast(Dict[str, List[Any]], events)
            return lst  # dict of dict containing list of events
        # Only collect the specified event type
        if event_type == SCALARS:
            getter = event_acc.Scalars
        elif event_type == TENSORS:
            getter = event_acc.Tensors
        elif event_type == HISTOGRAMS:
            getter = event_acc.Histograms
        else:
            raise KeyError(f"Unknown event_type: {event_type}")
        if tag is not None:
            return getter(tag)  # list of events
        ret = {}
        for t in event_acc.Tags()[event_type]:
            ret[t] = getter(t)
        return ret  # dict containing list of events

    @staticmethod
    def _make_empty_dict(data) -> Dict[str, Any]:
        """Generate a dictionary containing an empty list for each event type.

        :return: A dictionary containing an empty list for each event type.
        :rtype: Dict[str, Any]
        """
        return {
            # IMAGES: [],
            # AUDIO: [],
            HISTOGRAMS: copy.copy(data),
            SCALARS: copy.copy(data),
            # COMPRESSED_HISTOGRAMS: [],
            TENSORS: copy.copy(data),
            # GRAPH: [],
            # META_GRAPH: [],
            # RUN_METADATA: [],
            # HPARAMS: [],
        }

    def __repr__(self) -> str:
        """Returns the string representation of the `SummaryWriter` instance.
        Should be invoked by `repr(reader)`.

        :return: The string representation of the `SummaryWriter` instance.
        :rtype: str
        """
        return f"SummaryReader(log_path='{self.log_path}')"

    def __getitem__(self, child_idx) -> 'SummaryReader':
        """Returns the child `SummaryReader` with index `child_idx`. Should
        be invoked by `reader[idx]`.

        :return: The child `SummaryReader` with index `child_idx`.
        :rtype: SummaryReader
        """
        return self.children[child_idx]
