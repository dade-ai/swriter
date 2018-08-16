# -*- coding: utf-8 -*-
import os
from tensorflow.core.util import event_pb2
from tensorflow.python import pywrap_tensorflow
from tensorflow.python.platform import gfile
from tensorflow.python.util import compat
import six

from multiprocessing import Queue
import threading
import time


def create_writer(logdir, filename_suffix):
    if not gfile.IsDirectory(logdir):
        gfile.MakeDirs(logdir)
    ev_writer = pywrap_tensorflow.EventsWriter(
        compat.as_bytes(os.path.join(logdir, "events")))
    if filename_suffix:
        ev_writer.InitWithSuffix(compat.as_bytes(filename_suffix))
    return ev_writer


class EventWriters(object):
    """
    modified from tf.summary.EventFileWriter
    EventFileWriter alternative to write in multi `logdirs`
    """

    def __init__(self, logdir, max_queue=10, flush_secs=120, filename_suffix=None):
        self._logdir = logdir
        if not gfile.IsDirectory(self._logdir):
            gfile.MakeDirs(self._logdir)
        self._event_queue = six.moves.queue.Queue(max_queue)
        # self._root_writer = create_writer(logdir, filename_suffix)
        self._ev_writers = dict()
        self._flush_secs = flush_secs
        self._sentinel_event = self._get_sentinel_event()
        self._filename_suffix = filename_suffix
        self._closed = False
        # run worker thread
        self._worker = _EventLoggerThread(self._event_queue, self,
                                          self._flush_secs, self._sentinel_event)

        self._worker.start()

    def _get_sentinel_event(self):
        """Generate a sentinel event for terminating worker."""
        return event_pb2.Event()

    def get_real_writer(self, key):
        # if key is None:
        #     return self._root_writer
        writer = self._ev_writers.get(key, None)
        if writer is None:
            logdir = os.path.join(self._logdir, key)
            writer = create_writer(logdir, self._filename_suffix)
            self._ev_writers[key] = writer
        return writer

    def get_logdir(self):
        """Returns the directory where event file will be written."""
        return self._logdir

    def reopen(self):
        """Reopens the EventFileWriter.

        Can be called after `close()` to add more events in the same directory.
        The events will go into a new events file.

        Does nothing if the EventFileWriter was not closed.
        """
        if self._closed:
            self._worker = _EventLoggerThread(self._event_queue, self,
                                          self._flush_secs, self._sentinel_event)
            self._worker.start()
            self._closed = False

    def add_event(self, event):
        """Adds an event to the event file.

        Args:
        event: An `Event` protocol buffer.
        """
        if not self._closed:
            self._event_queue.put(event)

    def flush(self):
        self._event_queue.join()
        for k, writer in self._ev_writers.items():
            writer.Flush()

    def close(self):
        """Flushes the event file to disk and close the file.

        Call this method when you do not need the summary writer anymore.
        """
        self.add_event(self._sentinel_event)
        self.flush()
        self._worker.join()
        for k, writer in self._ev_writers.items():
            writer.Close()
        self._closed = True

    def WriteEvent(self, event):
        try:
            key, realevent = event
        except AttributeError:
            key = ''
            # if isinstance(event, Event):
            realevent = event
        writer = self.get_real_writer(key)
        writer.WriteEvent(realevent)

    def Flush(self):
        self.flush()


class _EventLoggerThread(threading.Thread):
    """Thread that logs events."""

    def __init__(self, queue, ev_writers, flush_secs, sentinel_event):
        """Creates an _EventLoggerThread.

        Args:
          queue: A Queue from which to dequeue events.
          ev_writer: An event writer. Used to log brain events for
           the visualizer.
          flush_secs: How often, in seconds, to flush the
            pending file to disk.
          sentinel_event: A sentinel element in queue that tells this thread to
            terminate.
        """
        threading.Thread.__init__(self)
        self.daemon = True
        self._queue = queue
        self._ev_writers = ev_writers
        self._flush_secs = flush_secs
        # The first event will be flushed immediately.
        self._next_event_flush_time = 0
        self._sentinel_event = sentinel_event

    def run(self):
        while True:
            event = self._queue.get()
            if event is self._sentinel_event:
                self._queue.task_done()
                break
            try:
                self._ev_writers.WriteEvent(event)
                # Flush the event writer every so often.
                now = time.time()
                if now > self._next_event_flush_time:
                    self._ev_writers.Flush()
                    # Do it again in two minutes.
                    self._next_event_flush_time = now + self._flush_secs
            finally:
                self._queue.task_done()
