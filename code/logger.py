import tensorflow as tf
import numpy as np


class Logger(object):
    class Loggable(object):
        def __init__(self, summary, console_str):
            self.summary = summary
            self.console_str = console_str

        def to_tensorboard(self, filewriter, itr):
            filewriter.add_summary(self.summary, global_step=itr)
            return self

        def to_console(self, itr, append=''):
            print('{}: {} {}'.format(itr, self.console_str, append))

    class Numpy1DFormatter(object):
        def __init__(self, wrapper_str='{}', max_elements=None, precision=3, sep=','):
            self._array2string = lambda _arr: wrapper_str.format(
                    np.array2string(
                            _arr.flatten()[:max_elements],
                            precision=precision, separator=sep))

        def format(self, arr):
            return self._array2string(arr)


    def __init__(self):
        self._summaries = []
        self._summary = None  # merged
        self._sess = None
        self._console_format_strs = []
        self._console_tensors = []
        self._final = False

    def add_summaries(self, summaries):
        assert isinstance(summaries, list)
        self._summaries.extend(summaries)

    def add_console_tensor(self, formatter, tensor):
        """
        :param formatter: object responding to -format(), e.g., a string with {}, or Numpy1DFormatter
        :param tensor:
        :return:
        """
        self._console_format_strs.append(formatter)
        self._console_tensors.append(tensor)

    def finalize_with_sess(self, sess):
        assert not self._final
        self._final = True
        self._sess = sess
        assert len(self._console_format_strs) == len(self._console_tensors)
        self._summary = tf.summary.merge(self._summaries)

    def log(self):
        summary, console_tensor = self._sess.run([self._summary, self._console_tensors])
        return Logger.Loggable(
            summary=summary,
            console_str=', '.join(
                format_str.format(tensor)
                for format_str, tensor in zip(self._console_format_strs, console_tensor)))


