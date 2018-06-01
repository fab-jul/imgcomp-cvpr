import tensorflow as tf
import os
from tensorflow.contrib import slim as slim
import numpy as np

from collections import namedtuple
from contextlib import contextmanager


import quantizer


# returned by _Network.encode
# z is the bottleneck before quantization
EncoderOutput = namedtuple('EncoderOutput', ['qbar', 'qhard', 'symbols', 'z', 'heatmap'])

# returned by _Network._quantize
_QuantizerOutput = namedtuple('_QuantizerOutput', ['qbar', 'qsoft', 'qhard', 'symbols'])


SCOPE_AE = 'autoencoder'
SCOPE_AE_ENC = SCOPE_AE + '/encoder'
SCOPE_AE_DEC = SCOPE_AE + '/decoder'


def get_network_cls(config):
    return {
        'CVPR': _CVPR,
    }[config.arch]



class _Network(object):
    def __init__(self, config, quantize=True):
        self.config = config
        self.quantize = quantize

        self.reuse_enc = False
        self.reuse_dec = False

        self.num_chan_bn_including_heatmap = config.num_chan_bn + 1

        self._centers = None  # Set in encode(); Access with get_centers_variable

    @staticmethod
    def get_subsampling_factor():
        """ Overridden by subclasses """
        raise NotImplementedError()

    def encode(self, x, is_training):  # -> EncoderOutput instance, with qbar, qhard, heatmap
        assert tf.float32.is_compatible_with(x.dtype), 'Expected float32 for x, got {}'.format(x.dtype)
        with self._building_ctx(SCOPE_AE_ENC, self.reuse_enc):
            if self._centers is None and self.quantize:
                self._centers = quantizer.create_centers_variable(self.config)
                quantizer.create_centers_regularization_term(self.config, self._centers)

            self.reuse_enc = True
            return self._encode(x, is_training)

    def decode(self, q, is_training):  # -> x_out
        with self._building_ctx(SCOPE_AE_DEC, self.reuse_dec):
            self.reuse_dec = True
            return self._decode(q, is_training)

    def get_centers_variable(self):
        if self._centers is None:
            raise ValueError('Call -encode(...) before trying to access centers')
        return self._centers

    @staticmethod
    def encoder_variables():
        """ Includes center variable """
        return _Network._get_trainable_vars_assert_non_empty(scope=SCOPE_AE_ENC)

    @staticmethod
    def decoder_variables():
        return _Network._get_trainable_vars_assert_non_empty(scope=SCOPE_AE_DEC)

    @staticmethod
    def encoder_regularization_loss():
        """ includes centers regularization """
        return tf.losses.get_regularization_loss(scope=SCOPE_AE_ENC)

    @staticmethod
    def decoder_regularization_loss():
        return tf.losses.get_regularization_loss(scope=SCOPE_AE_DEC)

    # ------------------------------------------------------------------------------

    def _encode(self, x, is_training):
        """ Overridden by subclasses """
        raise NotImplementedError()

    def _decode(self, q, is_training):
        """ Overridden by subclasses """
        raise NotImplementedError()

    @contextmanager
    def _building_ctx(self, scope_name, reuse):
        with tf.variable_scope(scope_name, reuse=reuse):
            with slim.arg_scope([slim.conv2d, slim.conv2d_transpose, residual_block],
                                weights_regularizer=slim.l2_regularizer(self.config.regularization_factor),
                                data_format='NCHW'):
                yield

    @contextmanager
    def _batch_norm_scope(self, is_training):
        batch_norm_params = self._batch_norm_params(is_training)
        with slim.arg_scope([slim.conv2d, slim.conv2d_transpose],
                            normalizer_fn=slim.batch_norm,
                            normalizer_params=batch_norm_params):
            with slim.arg_scope([slim.batch_norm], **batch_norm_params):
                yield

    @staticmethod
    def _batch_norm_params(is_training):
        return {
            'decay': 0.9,
            'epsilon': 1e-5,
            'scale': True,
            'updates_collections': tf.GraphKeys.UPDATE_OPS,
            'fused': True,
            'is_training': is_training,
            'data_format': 'NCHW',
        }

    def _quantize(self, inputs):
        if not self.quantize:
            return _QuantizerOutput(inputs, None, None, None)
        assert self._centers is not None
        qsoft, qhard, symbols = quantizer.quantize(inputs, self._centers, sigma=1)
        with tf.name_scope('qbar'):
            qbar = qsoft + tf.stop_gradient(qhard - qsoft)
        return _QuantizerOutput(qbar, qsoft, qhard, symbols)

    def _normalize(self, data):
        with tf.name_scope('normalize'):
            style = self.config.normalization
            if style == 'OFF':
                return data
            if style == 'FIXED':
                mean, var = self._get_mean_var()
                return (data - mean) / np.sqrt(var + 1e-10)
            raise ValueError('Invalid normalization style {}'.format(style))

    def _denormalize(self, data):
        with tf.name_scope('denormalize'):
            style = self.config.normalization
            if style == 'OFF':
                return data
            if style == 'FIXED':
                mean, var = self._get_mean_var()
                return (data * np.sqrt(var + 1e-10)) + mean
            raise ValueError('Invalid normalization style {}'.format(style))

    @staticmethod
    def _clip_to_image_range(x):
        return tf.clip_by_value(x, 0, 255, name='clip')

    @staticmethod
    def _get_mean_var():
        mean = np.array([121.85369873, 113.58860779, 100.63715363], dtype=np.float32)
        var = np.array([4746.37695312, 4454.13964844, 4812.234375], dtype=np.float32)

        # make mean, var into (3, 1, 1) so that they broadcast with NCHW
        mean = np.expand_dims(np.expand_dims(mean, -1), -1)
        var = np.expand_dims(np.expand_dims(var, -1), -1)

        return mean, var

    @staticmethod
    def _get_heatmap3D(bottleneck):
        """
        create heatmap3D, where
            heatmap3D[x, y, c] = heatmap[x, y] - c \intersect [0, 1]
        """
        assert bottleneck.shape.ndims == 4, bottleneck.shape

        with tf.name_scope('heatmap'):
            C = int(bottleneck.shape[1]) - 1  # -1 because first channel is heatmap

            heatmap_channel = bottleneck[:, 0, :, :]  # NHW
            heatmap2D = tf.nn.sigmoid(heatmap_channel) * C  # NHW
            c = tf.range(C, dtype=tf.float32)  # C

            # reshape heatmap2D for broadcasting
            heatmap = tf.expand_dims(heatmap2D, 1)  # N1HW
            # reshape c for broadcasting
            c = tf.reshape(c, (C, 1, 1))  # C11

            # construct heatmap3D
            # if heatmap[x, y] == C, then heatmap[x, y, c] == 1 \forall c \in {0, ..., C-1}
            heatmap3D = tf.maximum(tf.minimum(heatmap - c, 1), 0, name='heatmap3D')  # NCHW
            return heatmap3D

    @staticmethod
    def _mask_with_heatmap(bottleneck, heatmap3D):
        with tf.name_scope('heatmap_mask'):
            bottleneck_without_heatmap = bottleneck[:, 1:, ...]
            return heatmap3D * bottleneck_without_heatmap

    @staticmethod
    def _get_trainable_vars_assert_non_empty(scope):
        trainable_vars = tf.trainable_variables(scope)
        assert len(trainable_vars) > 0, 'No trainable variables found in scope {}. All: {}'.format(
            scope, tf.trainable_variables())
        return trainable_vars


arch_param_n = 128


class _CVPR(_Network):
    @staticmethod
    def get_subsampling_factor():
        return 8

    def _encode(self, x, is_training):
        n = arch_param_n
        with self._batch_norm_scope(is_training):
            net = self._normalize(x)
            net = slim.conv2d(net, n // 2, [5, 5], stride=2, scope='h1')
            net = slim.conv2d(net, n, [5, 5], stride=2, scope='h2')
            residual_input_0 = net
            for b in range(self.config.arch_param_B):
                residual_input_b = net
                with tf.variable_scope('res_block_enc_{}'.format(b)):
                    net = residual_block(net, n, num_conv2d=2, kernel_size=[3, 3], scope='enc_{}_1'.format(b))
                    net = residual_block(net, n, num_conv2d=2, kernel_size=[3, 3], scope='enc_{}_2'.format(b))
                    net = residual_block(net, n, num_conv2d=2, kernel_size=[3, 3], scope='enc_{}_3'.format(b))
                net = net + residual_input_b
            net = residual_block(net, n, num_conv2d=2, kernel_size=[3, 3], scope='res_block_enc_final',
                                 activation_fn=None)
            net = net + residual_input_0
            # BN
            C = self.num_chan_bn_including_heatmap if self.config.heatmap else self.config.num_chan_bn
            net = slim.conv2d(net, C, [5, 5], stride=2, activation_fn=None, scope='to_bn')
            if self.config.heatmap:
                heatmap = self._get_heatmap3D(bottleneck=net)
                net = self._mask_with_heatmap(net, heatmap)
            else:
                heatmap = None
            qout = self._quantize(net)
            return EncoderOutput(qout.qbar, qout.qhard, qout.symbols, net, heatmap)

    def _decode(self, q, is_training):
        with self._batch_norm_scope(is_training):
            n = arch_param_n
            fa = 3
            fb = 5
            net = slim.conv2d_transpose(q, n, [fa, fa], stride=2, scope='from_bn')
            residual_input_0 = net
            for b in range(self.config.arch_param_B):
                residual_input_b = net
                with tf.variable_scope('res_block_dec_{}'.format(b)):
                    net = residual_block(net, n, num_conv2d=2, kernel_size=[3, 3], scope='dec_{}_1'.format(b))
                    net = residual_block(net, n, num_conv2d=2, kernel_size=[3, 3], scope='dec_{}_2'.format(b))
                    net = residual_block(net, n, num_conv2d=2, kernel_size=[3, 3], scope='dec_{}_3'.format(b))
                net = net + residual_input_b
            net = residual_block(net, n, num_conv2d=2, kernel_size=[3, 3], scope='dec_after_res',
                                 activation_fn=None)
            net = net + residual_input_0

            net = slim.conv2d_transpose(net, n // 2, [fb, fb], stride=2, scope='h12')
            net = slim.conv2d_transpose(net, 3, [fb, fb], stride=2, scope='h13', activation_fn=None)
            net = self._denormalize(net)
            net = self._clip_to_image_range(net)
            return net


# ------------------------------------------------------------------------------


@slim.add_arg_scope
def residual_block(x, num_outputs, num_conv2d, **kwargs):
    assert 'num_outputs' not in kwargs
    kwargs['num_outputs'] = num_outputs

    residual_input = x
    with tf.variable_scope(kwargs.get('scope', None), 'res'):
        for conv_i in range(num_conv2d):
            kwargs['scope'] = 'conv{}'.format(conv_i + 1)
            if conv_i == (num_conv2d - 1):  # no relu after final conv
                kwargs['activation_fn'] = None
            x = slim.conv2d(x, **kwargs)

        return x + residual_input

