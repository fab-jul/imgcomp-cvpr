import tensorflow as tf
import numpy as np

from contextlib import contextmanager
from fjcommon import functools_ext
from fjcommon import tf_helpers


def get_network_cls(pc_config):
    """ Returns a class that is a subclass of _Network. """
    return {
        'res_shallow': _ResShallow,
    }[pc_config.arch]


def context_shape_from_context_size(context_size):
    """ :return context shape as DHW """
    return context_size // 2 + 1, context_size, context_size


def context_size_from_context_shape(context_shape):
    return context_shape[-1]


class _Network3D(object):
    _PROBCLASS_SCOPE = 'probclass3d'

    def __init__(self, pc_config, num_centers):
        """
        :param pc_config: Expected to contain
                - kernel_size: int
                - arch: str, see get_network_cls
        """
        self.config = pc_config
        self.reuse = False
        self.L = num_centers

        self.first_mask = None
        self.other_mask = None

    @classmethod
    def get_num_layers(cls):
        raise NotImplementedError()

    @classmethod
    def get_context_size(cls, config):
        """
        width / height of the receptive field
        """
        return cls.get_num_layers() * (config.kernel_size - 1) + 1

    def auto_pad_value(self, ae):
        return (0 if not self.config.use_centers_for_padding else
                ae.get_centers_variable()[0])

    def bitcost(self, q, target_symbols, is_training, pad_value=0):
        """
        Pads q, creates PC network, calculates cross entropy between output of PC network and target_symbols
        :param q: NCHW
        :param target_symbols:
        :param is_training:
        :return: bitcost: NCHW
        """
        tf_helpers.assert_ndims(q, 4)

        with self._building_ctx(self.reuse):
            if self.first_mask is None:
                self.first_mask = self.create_first_mask()  # DHWio
                self.other_mask = self.create_other_mask()  # DHWio

            self.reuse = True

            targets_one_hot = tf.one_hot(target_symbols, depth=self.L, axis=-1, name='target_symbols')

            q_pad = pad_for_probclass3d(
                    q, context_size=self.get_context_size(self.config),
                    pad_value=pad_value, learn_pad_var=False)
            with tf.variable_scope('logits'):
                # make it into NCHWT, where T is the channel dim of the conv3d
                q_pad = tf.expand_dims(q_pad, -1, name='NCHWT')
                logits = self._logits(q_pad, is_training)

            if self.config.regularization_factor is not None:
                print('Creating PC regularization...')
                weights = _get_all_conv3d_weights_in_scope(self._PROBCLASS_SCOPE)
                assert len(weights) > 0
                reg = self.config.regularization_factor * tf.add_n(list(map(tf.nn.l2_loss, weights)))
                tf.losses.add_loss(reg, tf.GraphKeys.REGULARIZATION_LOSSES)

            if targets_one_hot.shape.is_fully_defined() and logits.shape.is_fully_defined():
                tf_helpers.assert_equal_shape(targets_one_hot, logits)

            with tf.name_scope('bitcost'):
                # softmax_cross_entropy_with_logits is basis e, change base to 2
                log_base_change_factor = tf.constant(np.log2(np.e), dtype=tf.float32)
                bc = tf.nn.softmax_cross_entropy_with_logits(
                    logits=logits, labels=targets_one_hot) * log_base_change_factor  # NCHW

            return bc


    def variables(self):
        trainable_vars = tf.trainable_variables(self._PROBCLASS_SCOPE)
        assert len(trainable_vars) > 0, 'No trainable variables found in scope {}. All: {}'.format(
            self._PROBCLASS_SCOPE, tf.trainable_variables())
        return trainable_vars

    def regularization_loss(self):
        if self.config.regularization_factor is None:
            return None
        print('Regularization loss!')
        return tf.losses.get_regularization_loss(scope=self._PROBCLASS_SCOPE)

    @contextmanager
    def _building_ctx(self, reuse):
        with tf.variable_scope(self._PROBCLASS_SCOPE, reuse=reuse):
            yield

    def get_network_variables(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                 scope=self._PROBCLASS_SCOPE)

    def _logits(self, q, is_training):
        raise NotImplementedError()

    @property
    def filter_shape(self):
        K = self.config.kernel_size
        return K // 2 + 1, K, K  # CHW

    def create_first_mask(self):
        with tf.name_scope('first_mask'):
            K = self.config.kernel_size
            # mask is DHW
            mask = np.ones(self.filter_shape, dtype=np.float32)
            # zero out D=1,
            # - everything to the right of the central pixel, including the central pixel
            mask[-1, K // 2, K // 2:] = 0
            # - all rows below the central row
            mask[-1, K // 2 + 1:, :] = 0

            mask = np.expand_dims(np.expand_dims(mask, -1), -1)  # Make into DHWio, for broadcasting with 3D filters
            return _Network3D._make_tf_conv3d_mask(mask)

    def create_other_mask(self):
        with tf.name_scope('other_mask'):
            K = self.config.kernel_size
            # mask is DHW
            mask = np.ones(self.filter_shape, dtype=np.float32)
            # zero out D=1,
            # - everything to the right of the central pixel, except the central pixel
            mask[-1, K // 2, K // 2 + 1:] = 0
            # - all rows below the central row
            mask[-1, K // 2 + 1:, :] = 0

            mask = np.expand_dims(np.expand_dims(mask, -1), -1)  # Make into DHWio, for broadcasting with 3D filters
            return _Network3D._make_tf_conv3d_mask(mask)

    @staticmethod
    def _make_tf_conv3d_mask(mask):
        assert mask.ndim == 5, 'Expected DHWio'
        mask = tf.constant(mask)
        mask = tf.stop_gradient(mask)
        return mask

    def residual_block(self, x, num_conv2d=2, name=None):
        num_outputs = x.shape.as_list()[-1]
        residual_input = x
        activation_fn = tf.nn.relu
        with tf.variable_scope(name, 'res'):
            for conv_i in range(num_conv2d):
                if conv_i == (num_conv2d - 1):  # no relu after final conv
                    activation_fn = None
                x = conv3d('conv{}'.format(conv_i + 1),
                           x, num_outputs, self.filter_shape, self.other_mask,
                           activation_fn=activation_fn)
            return x + residual_input[..., 2:, 2:-2, 2:-2, :]  # for padding


class _ResShallow(_Network3D):
    """
    supported parameters:
        arch_param__k:      number of channels for the conv layers
        arch_param__non_linearity :: 'relu', 'tanh', None:
                            type of non-linearity before output
    """
    _NUM_RESIDUAL = 1

    @classmethod
    def get_num_layers(cls):
        num_conv = 2
        per_residual = 2
        return num_conv + _ResShallow._NUM_RESIDUAL * per_residual

    def _logits(self, q, is_training):
        k = self.config.arch_param__k
        net = q
        net = conv3d('conv0', net, k, self.filter_shape, filter_mask=self.first_mask)
        for res_i in range(self._NUM_RESIDUAL):
            net = self.residual_block(net, name='res{}'.format(res_i + 1))
        net = conv3d('conv2', net, self.L, self.filter_shape, filter_mask=self.other_mask)
        return net


# Helpers ----------------------------------------------------------------------


def conv3d(name,
           x,  # NCHWD
           num_outputs,
           filter_shape,  # (C, H, W)
           filter_mask=None,
           strides=None,
           activation_fn=tf.nn.relu,
           padding='VALID',
           weights_initializer=tf.contrib.layers.xavier_initializer(),
           biases_initializer=tf.zeros_initializer(),
           ):
    assert name is not None, 'Need name'
    assert len(filter_shape) == 3
    if not strides:
        strides = [1, 1, 1, 1, 1]
    if not activation_fn:
        activation_fn = functools_ext.identity

    num_inputs = x.shape.as_list()[-1]
    filter_shape = tuple(filter_shape) + (num_inputs, num_outputs)

    masked = filter_mask is not None
    scope_name = 'conv3d_{}'.format(name) + ('_mask' if masked else '')
    with tf.variable_scope(scope_name):
        weights = tf.get_variable('weights', shape=filter_shape, dtype=tf.float32,
                                  initializer=weights_initializer)
        if filter_mask is not None:
            weights = weights * filter_mask

        biases = tf.get_variable('biases', shape=(num_outputs,), dtype=tf.float32,
                                 initializer=biases_initializer)
        out = tf.nn.conv3d(x, weights, strides, padding, name='conv3d')
        out = tf.nn.bias_add(out, biases, name='bias3d')
        out = activation_fn(out)
        return out


def _get_all_conv3d_weights_in_scope(scope):
    return [v for v in tf.trainable_variables(scope=scope) if 'weights' in v.name]


def pad_for_probclass3d(x, context_size, pad_value=0, learn_pad_var=False):
    """
    :param x: NCHW tensorflow Tensor
    """
    with tf.name_scope('pad_cs' + str(context_size)):
        pad = context_size // 2
        assert pad >= 1
        if learn_pad_var:
            if not isinstance(pad_value, tf.Variable):
                print('Warn: Expected tf.Variable for padding, got {}'.format(pad_value))
            return pc_pad_grad(x, pad, pad_value)

        pads = [[0, 0],  # don't pad batch dimension
                [pad, 0],  # don't pad depth_future, it's not seen by any filter
                [pad, pad],
                [pad, pad]]
        assert len(pads) == _get_ndims(x)
        return tf.pad(x, pads, constant_values=pad_value)


def pc_pad_grad(x, pad, pad_var):
    """
    Like tf.pad but gradients flow to pad_var
    :param x: NCHW
    :param pad: will pad with
        pads = [[0, 0],     don't pad batch dimension
                [pad, 0],   don't pad depth_future, it's not seen by any filter
                [pad, pad],
                [pad, pad]]
    :param pad_var: value to use for padding
    :return:
    """
    with tf.name_scope('pc_pad_grad'):
        n, c, h, w = x.shape.as_list()
        with tf.name_scope('pad_var_NCHW'):
            pad_var = tf.expand_dims(tf.expand_dims(tf.expand_dims(tf.expand_dims(pad_var, 0), 0), 0), 0)

        with tf.name_scope('front'):
            front = tf.tile(pad_var, [n, pad, h, w])
            x_front = tf.concat((front, x), 1)

        with tf.name_scope('left_right'):
            left = tf.tile(pad_var, [n, c + pad, h, pad])
            right = left
            x_front_left_right = tf.concat((left, x_front, right), 3)

        with tf.name_scope('top_bottom'):
            top = tf.tile(pad_var, [n, c + pad, pad, w + 2 * pad])
            bottom = top
            x_fron_left_right_top_bottom = tf.concat((top, x_front_left_right, bottom), 2)

        return x_fron_left_right_top_bottom


def undo_pad_for_probclass3d(x, context_size):
    """
    :param x: NCHW tensorflow Tensor or numpy array
    """
    with tf.name_scope('undo_pad_cs' + str(context_size)):
        pad = context_size // 2
        assert pad >= 1
        return x[:, pad:, pad:-pad, pad:-pad]


def _get_ndims(t_or_a):
    try:
        return t_or_a.shape.ndims
    except AttributeError:
        return t_or_a.ndim

