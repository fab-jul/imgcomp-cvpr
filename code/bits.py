import tensorflow as tf


def bitcost_to_bpp(bit_cost, input_batch):
    """
    :param bit_cost: NChw
    :param input_batch: N3HW
    :return: Chw / HW, i.e., num_bits / num_pixels
    """
    assert bit_cost.shape.ndims == input_batch.shape.ndims == 4, 'Expected NChw and N3HW, got {} and {}'.format(
        bit_cost, input_batch)
    with tf.name_scope('bitcost_to_bpp'):
        num_bits = tf.reduce_sum(bit_cost, name='num_bits')
        return num_bits / tf.to_float(num_pixels_in_input_batch(input_batch))


def num_pixels_in_input_batch(input_batch):
    assert int(input_batch.shape[1]) == 3, 'Expected N3HW, got {}'.format(input_batch)
    with tf.name_scope('num_pixels'):
        return tf.reduce_prod(tf.shape(input_batch)) / 3

