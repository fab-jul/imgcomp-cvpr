import tensorflow as tf
import functools
import itertools


# _MOMENTUM = 0.9

_EXAMPLE_CONFIG = """
lr_initial = 1e-4  # initial learning rate

constrain optimizer :: ADAM, MOMENTUM
optimizer = ADAM
optimizer_momentum = 0.9  # momentum to use if optimizer == MOMENTUM

constrain lr_schedule :: FIXED, DECAY
lr_schedule = DECAY
lr_schedule_decay_interval = 3  # num epochs before decay
lr_schedule_decay_rate = 0.1
"""


def create_learning_rate_tensor(config, input_pipeline, name):
    with tf.name_scope(name, default_name='lr'):
        lr = tf.constant(config.lr_initial, tf.float32, name='lr_initial')
        if config.lr_schedule == 'FIXED':
            return lr
        if config.lr_schedule == 'DECAY':
            global_step = tf.train.get_or_create_global_step()
            num_itr_per_epoch = get_num_itr_per_epoch(input_pipeline)
            return tf.train.exponential_decay(
                    lr, global_step,
                    decay_steps=num_itr_per_epoch * config.lr_schedule_decay_interval,
                    decay_rate=config.lr_schedule_decay_rate,
                    staircase=config.lr_schedule_decay_staircase)
        raise ValueError('Invalid lr_schedule {}'.format(config.lr_schedule))


def create_optimizer(config, learning_rate_tensor, name=None):
    return optimizer_cls(config)(learning_rate=learning_rate_tensor, name=name)


def optimizer_cls(config):
    return {
        'ADAM': tf.train.AdamOptimizer,
        'SGD': tf.train.GradientDescentOptimizer,
        'MOMENTUM': functools.partial(tf.train.MomentumOptimizer,
                                      momentum=config.optimizer_momentum, use_nesterov=True),
    }[config.optimizer]


def get_num_itr_per_epoch(input_pipeline):
    num_crops_per_img = input_pipeline.num_crops_per_img
    dataset = input_pipeline.dataset
    batch_size = input_pipeline.batch_size

    num_unique_imgs_per_batch = batch_size // num_crops_per_img
    num_training_imgs = dataset.num_images
    num_itr_per_epoch = num_training_imgs // num_unique_imgs_per_batch

    return num_itr_per_epoch
