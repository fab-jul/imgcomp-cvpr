import tensorflow as tf
from fjcommon import tf_helpers


_HARD_SIGMA = 1e7


# ------------------------------------------------------------------------------


def create_centers_variable(config):  # (C, L) or (L,)
    assert config.num_centers is not None
    return tf.get_variable(
        'centers', shape=(config.num_centers,), dtype=tf.float32,
        initializer=_get_centers_initializer(config))


def create_centers_regularization_term(config, centers):
    # Add centers regularization
    if config.regularization_factor_centers != 0:
        with tf.name_scope('centers/Regularizer'):  # following slim's naming convention
            reg = tf.to_float(config.regularization_factor_centers)
            centers_reg = tf.identity(reg * tf.nn.l2_loss(centers), name='l2_regularizer')
            tf.losses.add_loss(centers_reg, tf.GraphKeys.REGULARIZATION_LOSSES)



def _get_centers_initializer(config):
    minval, maxval = map(int, config.centers_initial_range)
    #return tf.linspace(tf.to_float(minval), maxval, config.num_centers)
    return tf.random_uniform_initializer(minval=minval, maxval=maxval, seed=666)


# ------------------------------------------------------------------------------


def quantize(x, centers, sigma):
    """ :return qsoft, qhard, symbols """
    with tf.name_scope('quantize'):
        return _quantize1d(x, centers, sigma, data_format='NCHW')


def _quantize1d(x, centers, sigma, data_format):
    """
    :return: (softout, hardout, symbols_vol)
        each of same shape as x, softout, hardout will be float32, symbols_vol will be int64
    """
    assert tf.float32.is_compatible_with(x.dtype), 'x should be float32'
    assert tf.float32.is_compatible_with(centers.dtype), 'centers should be float32'
    assert len(x.get_shape()) == 4, 'x should be NCHW or NHWC, got {}'.format(x.get_shape())
    assert len(centers.get_shape()) == 1, 'centers should be (L,), got {}'.format(centers.get_shape())

    if data_format == 'NHWC':
        x_t = tf_helpers.transpose_NHWC_to_NCHW(x)
        softout, hardout, symbols_hard = _quantize1d(x_t, centers, sigma, data_format='NCHW')
        return tuple(map(tf_helpers.transpose_NCHW_to_NHWC, (softout, hardout, symbols_hard)))

    # Note: from here on down, x is NCHW ---

    # count centers
    num_centers = centers.get_shape().as_list()[-1]

    with tf.name_scope('reshape_BCm1'):
        # reshape (B, C, w, h) to (B, C, m=w*h)
        x_shape_BCwh = tf.shape(x)
        B = x_shape_BCwh[0]  # B is not necessarily static
        C = int(x.shape[1])  # C is static
        x = tf.reshape(x, [B, C, -1])

        # make x into (B, C, m, 1)
        x = tf.expand_dims(x, axis=-1)

    with tf.name_scope('dist'):
        # dist is (B, C, m, L), contains | x_i - c_j | ^ 2
        dist = tf.square(tf.abs(x - centers))

    with tf.name_scope('phi_soft'):
        # (B, C, m, L)
        phi_soft = tf.nn.softmax(-sigma       * dist, dim=-1)
    with tf.name_scope('phi_hard'):
        # (B, C, m, L) probably not necessary due to the argmax!
        phi_hard = tf.nn.softmax(-_HARD_SIGMA * dist, dim=-1)

        symbols_hard = tf.argmax(phi_hard, axis=-1)
        phi_hard = tf.one_hot(symbols_hard, depth=num_centers, axis=-1, dtype=tf.float32)

    with tf.name_scope('softout'):
        softout = phi_times_centers(phi_soft, centers)
    with tf.name_scope('hardout'):
        hardout = phi_times_centers(phi_hard, centers)

    def reshape_to_BCwh(t_):
        with tf.name_scope('reshape_BCwh'):
            return tf.reshape(t_, x_shape_BCwh)
    return tuple(map(reshape_to_BCwh, (softout, hardout, symbols_hard)))


def phi_times_centers(phi, centers):
    matmul_innerproduct = phi * centers  # (B, C, m, L)
    return tf.reduce_sum(matmul_innerproduct, axis=3)  # (B, C, m)


