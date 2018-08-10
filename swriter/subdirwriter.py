# -*- coding: utf-8 -*-
from tensorflow.python.summary.writer.writer import FileWriter
from tensorflow.python.eager import context
from .eventwriters import EventWriters
import time


class SubdirWriter(FileWriter):

    def __init__(self,
                 logdir,
                 graph=None,
                 max_queue=10,
                 flush_secs=120,
                 graph_def=None,
                 filename_suffix=None):
        if context.executing_eagerly():
            raise RuntimeError("ManyWriter is not compatible with eager execution. "
                               "Use tf.contrib.summary instead.")
        event_writer = EventWriters(logdir, max_queue, flush_secs, filename_suffix)
        self._subdir_context = ''
        super(FileWriter, self).__init__(event_writer, graph, graph_def)

    def add_summary(self, summary, global_step=None, subdir=''):
        """Adds a `Summary` protocol buffer to the event file.

        This method wraps the provided summary in an `Event` protocol buffer
        and adds it to the event file.

        You can pass the result of evaluating any summary op, using
        @{tf.Session.run} or
        @{tf.Tensor.eval}, to this
        function. Alternatively, you can pass a `tf.Summary` protocol
        buffer that you populate with your own data. The latter is
        commonly done to report evaluation results in event files.

        Args:
        summary: A `Summary` protocol buffer, optionally serialized as a string.
        global_step: Number. Optional global step value to record with the
          summary.
        subdir: string subdir
        """
        self._subdir_context = subdir
        super(SubdirWriter, self).add_summary(summary, global_step=global_step)

    def add_session_log(self, session_log, global_step=None, subdir=''):
        self._subdir_context = subdir
        super(SubdirWriter, self).add_session_log(session_log, global_step=global_step)

    def _add_graph_def(self, graph_def, global_step=None, subdir=''):
        self._subdir_context = subdir
        super(SubdirWriter, self)._add_graph_def(graph_def, global_step=global_step)

    def add_graph(self, graph, global_step=None, graph_def=None, subdir=''):
        self._subdir_context = subdir
        super(SubdirWriter, self).add_graph(graph, global_step=global_step, graph_def=graph_def)

    def add_meta_graph(self, meta_graph_def, global_step=None, subdir=''):
        self._subdir_context = subdir
        super(SubdirWriter, self).add_meta_graph(meta_graph_def, global_step=global_step)

    def add_run_metadata(self, run_metadata, tag, global_step=None, subdir=''):
        self._subdir_context = subdir
        super(SubdirWriter, self).add_run_metadata(run_metadata, tag, global_step=global_step)

    def _add_event(self, event, step):
        event.wall_time = time.time()
        if step is not None:
            event.step = int(step)

        if self._subdir_context:
            subdir = self._subdir_context
        else:
            subdir = ''
        self._subdir_context = ''
        self.event_writer.add_event((subdir, event))
