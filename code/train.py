import fasteners
import tensorflow as tf
from fjcommon import tf_helpers
from fjcommon import config_parser
from fjcommon import functools_ext

import time
import os
import subprocess
import argparse

from constants import NUM_PREPROCESS_THREADS, NUM_CROPS_PER_IMG
from collections import namedtuple

from restore_manager import RestoreManager
from saver import Saver
import inputpipeline
import training_helpers
import probclass
import autoencoder
import logdir_helpers
import ms_ssim
import bits
from logger import Logger
import sheets_logger
import numpy as np
from codec_distance import CodecDistance, CodecDistanceReadException


# Enable TF logging output
tf.logging.set_verbosity(tf.logging.INFO)


_LOG_DIR_FORMAT = """
- LOG DIR ----------------------------------------------------------------------
{}
--------------------------------------------------------------------------------"""

_STARTING_TRAINING_INFO_STR = """
- STARTING TRAINING ------------------------------------------------------------"""


_MAX_METADATA_RUNS = 1  # if --log_run_metadata is given, how many times should run metadata be logged


_EPS = 1e-5


TrainFlags = namedtuple(
        'TrainFlags',
        ['log_run_metadata', 'log_interval_train', 'log_interval_test', 'log_interval_save', 'summarize_grads'])


Datasets = namedtuple('Datasets', ['train', 'test', 'codec_distance'])


# note that (train_autoencoder=True, train_probclass=False) => probclass is still used to calculate H
def train(autoencoder_config_path, probclass_config_path,
          restore_manager: RestoreManager,
          log_dir_root,
          datasets: Datasets,
          train_flags: TrainFlags,
          ckpt_interval_hours: float,
          description: str):
    ae_config, ae_config_rel_path = config_parser.parse(autoencoder_config_path)
    pc_config, pc_config_rel_path = config_parser.parse(probclass_config_path)
    print_configs(('ae_config', ae_config), ('pc_config', pc_config))

    continue_in_ckpt_dir = restore_manager and restore_manager.continue_in_ckpt_dir
    if continue_in_ckpt_dir:
        logdir = restore_manager.log_dir
    else:
        logdir = logdir_helpers.create_unique_log_dir(
                [ae_config_rel_path, pc_config_rel_path], log_dir_root,
                restore_dir=restore_manager.ckpt_dir if restore_manager else None)
    print(_LOG_DIR_FORMAT.format(logdir))

    if description:
        _write_to_sheets(logdir_helpers.log_date_from_log_dir(logdir),
                         ae_config_rel_path, pc_config_rel_path,
                         description,
                         git_ref=_get_git_ref(),
                         log_dir_root=log_dir_root,
                         is_continue=continue_in_ckpt_dir)

    ae_cls = autoencoder.get_network_cls(ae_config)
    pc_cls = probclass.get_network_cls(pc_config)

    # Instantiate autoencoder and probability classifier
    ae = ae_cls(ae_config)
    pc = pc_cls(pc_config, num_centers=ae_config.num_centers)

    # train ---
    ip_train = inputpipeline.InputPipeline(
            inputpipeline.get_dataset(datasets.train),
            ae_config.crop_size, batch_size=ae_config.batch_size,
            shuffle=False,
            num_preprocess_threads=NUM_PREPROCESS_THREADS, num_crops_per_img=NUM_CROPS_PER_IMG)
    x_train = ip_train.get_batch()

    enc_out_train = ae.encode(x_train, is_training=True)  # qbar is masked by the heatmap
    x_out_train = ae.decode(enc_out_train.qbar, is_training=True)
    # stop_gradient is beneficial for training. it prevents multiple gradients flowing into the heatmap.
    pc_in = tf.stop_gradient(enc_out_train.qbar)
    bc_train = pc.bitcost(pc_in, enc_out_train.symbols, is_training=True, pad_value=pc.auto_pad_value(ae))
    bpp_train = bits.bitcost_to_bpp(bc_train, x_train)
    d_train = Distortions(ae_config, x_train, x_out_train, is_training=True)
    # summing over channel dimension gives 2D heatmap
    heatmap2D = (tf.reduce_sum(enc_out_train.heatmap, 1) if enc_out_train.heatmap is not None
                 else None)

    # loss ---
    total_loss, H_real, pc_comps, ae_comps = get_loss(
            ae_config, ae, pc, d_train.d_loss_scaled, bc_train, enc_out_train.heatmap)
    train_op = get_train_op(ae_config, pc_config, ip_train, pc.variables(), total_loss)

    # test ---
    with tf.name_scope('test'):
        ip_test = inputpipeline.InputPipeline(
                inputpipeline.get_dataset(datasets.test),
                ae_config.crop_size, batch_size=ae_config.batch_size,
                num_preprocess_threads=NUM_PREPROCESS_THREADS, num_crops_per_img=1,
                big_queues=False, shuffle=False)
        x_test = ip_test.get_batch()
        enc_out_test = ae.encode(x_test, is_training=False)
        x_out_test = ae.decode(enc_out_test.qhard, is_training=False)
        bc_test = pc.bitcost(enc_out_test.qhard, enc_out_test.symbols,
                             is_training=False, pad_value=pc.auto_pad_value(ae))
        bpp_test = bits.bitcost_to_bpp(bc_test, x_test)
        d_test = Distortions(ae_config, x_test, x_out_test, is_training=False)

    try:  # Try to get codec distnace for current dataset
        codec_distance_ms_ssim = CodecDistance(datasets.codec_distance, codec='bpg', metric='ms-ssim')
        get_distance = functools_ext.catcher(
                ValueError, handler=functools_ext.const(np.nan), f=codec_distance_ms_ssim.distance)
        get_distance = functools_ext.compose(np.float32, get_distance)  # cast to float32
        d_BPG_test = tf.py_func(get_distance, [bpp_test, d_test.ms_ssim],
                                tf.float32, stateful=False, name='d_BPG')
        d_BPG_test.set_shape(())
    except CodecDistanceReadException as e:
        print('Cannot compute CodecDistance: {}'.format(e))
        d_BPG_test = tf.constant(np.nan, shape=(), name='ConstNaN')

    # ---

    train_logger = Logger()
    test_logger = Logger()
    distortion_name = ae_config.distortion_to_minimize

    train_logger.add_summaries(d_train.summaries_with_prefix('train'))
    # Visualize components of losses
    train_logger.add_summaries([
        tf.summary.scalar('train/PC_loss/{}'.format(name), comp) for name, comp in pc_comps])
    train_logger.add_summaries([
        tf.summary.scalar('train/AE_loss/{}'.format(name), comp) for name, comp in ae_comps])
    train_logger.add_summaries([tf.summary.scalar('train/bpp', bpp_train)])
    train_logger.add_console_tensor('loss={:.3f}', total_loss)
    train_logger.add_console_tensor('ms_ssim={:.3f}', d_train.ms_ssim)
    train_logger.add_console_tensor('bpp={:.3f}', bpp_train)
    train_logger.add_console_tensor('H_real={:.3f}', H_real)

    test_logger.add_summaries(d_test.summaries_with_prefix('test'))
    test_logger.add_summaries([
        tf.summary.scalar('test/bpp', bpp_test),
        tf.summary.scalar('test/distance_BPG_MS-SSIM', d_BPG_test),
        tf.summary.image('test/x_in', prep_for_image_summary(x_test, n=3, name='x_in')),
        tf.summary.image('test/x_out', prep_for_image_summary(x_out_test, n=3, name='x_out'))])
    if heatmap2D is not None:
        test_logger.add_summaries([
            tf.summary.image('test/hm', prep_for_grayscale_image_summary(heatmap2D, n=3, autoscale=True, name='hm'))])

    test_logger.add_console_tensor('ms_ssim={:.3f}', d_test.ms_ssim)
    test_logger.add_console_tensor('bpp={:.3f}', bpp_test)
    test_logger.add_summaries([
        tf.summary.histogram('centers', ae.get_centers_variable()),
        tf.summary.histogram('test/qbar', enc_out_test.qbar[:ae_config.batch_size//2, ...])])
    test_logger.add_console_tensor('d_BPG={:.6f}', d_BPG_test)
    test_logger.add_console_tensor(Logger.Numpy1DFormatter('centers={}'), ae.get_centers_variable())

    print('Starting session and queues...')
    with tf_helpers.start_queues_in_sess(init_vars=restore_manager is None) as (sess, coord):
        train_logger.finalize_with_sess(sess)
        test_logger.finalize_with_sess(sess)

        if restore_manager:
            restore_manager.restore(sess)

        saver = Saver(Saver.ckpt_dir_for_log_dir(logdir),
                      max_to_keep=1,
                      keep_checkpoint_every_n_hours=ckpt_interval_hours)

        train_loop(ae_config, sess, coord, train_op, train_logger, test_logger,
                   train_flags, logdir, saver, is_restored=restore_manager is not None)


def print_configs(*configs_with_names):
    print('\n---\n'.join('Using {}:\n{}'.format(name, config) for name, config in configs_with_names))


class _Timer(object):
    def __init__(self, log_interval, batch_size):
        self.log_interval = log_interval
        self.batch_size = batch_size
        self.start_time = time.time()

    def get_avg_ex_per_sec(self):
        avg_time_per_step = (time.time() - self.start_time) / self.log_interval
        avg_ex_per_sec = self.batch_size / avg_time_per_step
        return avg_ex_per_sec

    def reset(self):
        self.start_time = time.time()


def train_loop(
        config,
        sess,
        coord,
        train_op,
        train_logger: Logger,
        test_logger: Logger,
        train_flags: TrainFlags,
        log_dir,
        saver: Saver,
        is_restored=False):
    global_step = tf.train.get_or_create_global_step()

    job_id = logdir_helpers.log_date_from_log_dir(log_dir)
    fw = tf.summary.FileWriter(log_dir, graph=sess.graph)

    training_timer = _Timer(train_flags.log_interval_train, config.batch_size)
    itr = 0
    num_metadata_runs = 0

    if is_restored:
        itr = sess.run(global_step)
        train_logger.log().to_tensorboard(fw, itr).to_console(itr, append='Restored')
        test_logger.log().to_tensorboard(fw, itr).to_console(itr)

    print(_STARTING_TRAINING_INFO_STR)
    while not coord.should_stop():
        if (train_flags.log_run_metadata
                and num_metadata_runs < _MAX_METADATA_RUNS
                and (itr % (train_flags.log_interval_train - 1) == 0)):
            print('Logging run metadata...', end=' ')
            num_metadata_runs += 1
            (_, itr), run_metadata = run_and_fetch_metadata([train_op, global_step], sess)
            fw.add_run_metadata(run_metadata, str(itr), itr)
            print('Done')
        else:
            _, itr = sess.run([train_op, global_step])

        # Train Logging --
        if itr % train_flags.log_interval_train == 0:
            info_str = '(img/s: {:.1f}) {}'.format(training_timer.get_avg_ex_per_sec(), job_id)
            train_logger.log().to_tensorboard(fw, itr).to_console(itr, append=info_str)

        # Save --
        if itr % train_flags.log_interval_save == 0:
            print('Saving...')
            saver.save(sess, global_step)

        # Test Logging --
        if train_flags.log_interval_test > 0 and itr % train_flags.log_interval_test == 0:
            test_logger.log().to_tensorboard(fw, itr).to_console(itr)

        if itr % train_flags.log_interval_train == 0:  # Reset after all above for accurate timings
            training_timer.reset()


def run_and_fetch_metadata(fetches, sess):
    print('*** Adding metadata...')
    run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
    run_metadata = tf.RunMetadata()
    return sess.run(fetches, options=run_options, run_metadata=run_metadata), run_metadata


def prep_for_image_summary(t, n=3, autoscale=False, name='img'):
    """ given tensor t of shape NCHW, return t[:n, ...] transposed to NHWC, cast to uint8 """
    assert int(t.shape[1]) == 3, 'Expected N3HW, got {}'.format(t)
    with tf.name_scope('prep_' + name):
        t = tf_helpers.transpose_NCHW_to_NHWC(t[:n, ...])
        if autoscale:  # if t is float32, tf.summary.image will automatically rescale
            assert tf.float32.is_compatible_with(t.dtype)
            return t
        else:  # if t is uint8, tf.summary.image will NOT automatically rescale
            return tf.cast(t, tf.uint8, 'uint8')


def prep_for_grayscale_image_summary(t, n=3, autoscale=False, name='img'):
    assert len(t.shape) == 3
    with tf.name_scope('prep_' + name):
        t = t[:n, ...]
        t = tf.expand_dims(t, -1)  # NHW1
        if autoscale:
            assert tf.float32.is_compatible_with(t.dtype)
            return t
        else:
            return tf.cast(t, tf.uint8, name='uint8')


def get_loss(config, ae, pc, d_loss_scaled, bc, heatmap):
    assert config.H_target

    heatmap_enabled = heatmap is not None

    with tf.name_scope('losses'):
        bc_mask = (bc * heatmap) if heatmap_enabled else bc
        H_real = tf.reduce_mean(bc, name='H_real')
        H_mask = tf.reduce_mean(bc_mask, name='H_mask')
        H_soft = 0.5 * (H_mask + H_real)

        H_target = tf.constant(config.H_target, tf.float32, name='H_target')
        beta = tf.constant(config.beta, tf.float32, name='beta')

        pc_loss = beta * tf.maximum(H_soft - H_target, 0)

        # Adding Regularizers
        with tf.name_scope('regularization_losses'):
            reg_probclass = pc.regularization_loss()
            if reg_probclass is None:
                reg_probclass = 0
            reg_enc = ae.encoder_regularization_loss()
            reg_dec = ae.decoder_regularization_loss()
            reg_loss = reg_probclass + reg_enc + reg_dec

        pc_comps = [('H_mask',  H_mask),
                    ('H_real',  H_real),
                    ('pc_loss', pc_loss),
                    ('reg',     reg_probclass)]
        ae_comps = [('d_loss_scaled',     d_loss_scaled),
                    ('reg_enc_dec',       reg_enc + reg_dec)]

        total_loss = d_loss_scaled + pc_loss + reg_loss
        return total_loss, H_real, pc_comps, ae_comps


def get_train_op(ae_config, pc_config, input_pipeline_train, vars_probclass, total_loss):
    lr_ae = training_helpers.create_learning_rate_tensor(ae_config, input_pipeline_train, name='lr_ae')
    default_optimizer = training_helpers.create_optimizer(ae_config, lr_ae, name='Adam_AE')

    lr_pc = training_helpers.create_learning_rate_tensor(pc_config, input_pipeline_train, name='lr_pc')
    optimizer_pc = training_helpers.create_optimizer(pc_config, lr_pc, name='Adam_PC')

    special_optimizers_and_vars = [(optimizer_pc, vars_probclass)]

    return tf_helpers.create_train_op_with_different_lrs(
            total_loss, default_optimizer, special_optimizers_and_vars, summarize_gradients=False)


class Distortions(object):
    def __init__(self, config, x, x_out, is_training):
        assert tf.float32.is_compatible_with(x.dtype) and tf.float32.is_compatible_with(x_out.dtype)
        self.config = config

        with tf.name_scope('distortions_train' if is_training else 'distortions_test'):
            minimize_for = config.distortion_to_minimize
            assert minimize_for in ('mse', 'psnr', 'ms_ssim')
            # don't calculate MS-SSIM if not necessary to speed things up
            should_get_ms_ssim = minimize_for == 'ms_ssim'
            # if we don't minimize for PSNR, cast x and x_out to int before calculating the PSNR, because otherwise
            # PSNR is off. If not training, always cast to int, because we don't need the gradients.
            # equivalent for when we don't minimize for MSE
            cast_to_int_for_psnr = (not is_training) or minimize_for != 'psnr'
            cast_to_int_for_mse = (not is_training) or minimize_for != 'mse'
            self.mse = self.mean_over_batch(
                Distortions.get_mse_per_img(x, x_out, cast_to_int_for_mse), name='mse')
            self.psnr = self.mean_over_batch(
                Distortions.get_psnr_per_image(x, x_out, cast_to_int_for_psnr), name='psnr')
            self.ms_ssim = (
                Distortions.get_ms_ssim(x, x_out)
                if should_get_ms_ssim else None)

            with tf.name_scope('distortion_to_minimize'):
                self.d_loss_scaled = self._get_distortion_to_minimize(minimize_for)

    def summaries_with_prefix(self, prefix):
        return tf_helpers.list_without_None(
            tf.summary.scalar(prefix + '/mse', self.mse),
            tf.summary.scalar(prefix + '/psnr', self.psnr),
            tf.summary.scalar(prefix + '/ms_ssim', self.ms_ssim) if self.ms_ssim is not None else None)

    def _get_distortion_to_minimize(self, minimize_for):
        """ Returns a float32 that should be minimized in training. For PSNR and MS-SSIM, which increase for a
        decrease in distortion, a suitable factor is added. """
        if minimize_for == 'mse':
            return self.mse
        if minimize_for == 'psnr':
            return self.config.K_psnr - self.psnr
        if minimize_for == 'ms_ssim':
            return self.config.K_ms_ssim * (1 - self.ms_ssim)

        raise ValueError('Invalid: {}'.format(minimize_for))

    @staticmethod
    def mean_over_batch(d, name):
        assert len(d.shape) == 1, 'Expected tensor of shape (N,), got {}'.format(d.shape)
        with tf.name_scope('mean_' + name):
            return tf.reduce_mean(d, name='mean')

    @staticmethod
    def get_mse_per_img(inp, otp, cast_to_int):
        """
        :param inp: NCHW
        :param otp: NCHW
        :param cast_to_int: if True, both inp and otp are casted to int32 before the error is calculated,
        to ensure real world errors (image pixels are always quantized). But the error is always casted back to
        float32 before a mean per image is calculated and returned
        :return: float32 tensor of shape (N,)
        """
        with tf.name_scope('mse_{}'.format('int' if cast_to_int else 'float')):
            if cast_to_int:
                # Values are expected to be in 0...255, i.e., uint8, but tf.square does not support uint8's
                inp, otp = tf.cast(inp, tf.int32), tf.cast(otp, tf.int32)
            squared_error = tf.square(otp - inp)
            squared_error_float = tf.to_float(squared_error)
            mse_per_image = tf.reduce_mean(squared_error_float, axis=[1, 2, 3])
            return mse_per_image

    @staticmethod
    def get_psnr_per_image(inp, otp, cast_to_int):
        with tf.name_scope('psnr_{}'.format('int' if cast_to_int else 'float')):
            mse_per_image = Distortions.get_mse_per_img(inp, otp, cast_to_int)
            psnr_per_image = 10 * tf_helpers.log10(255.0 * 255.0 / mse_per_image)
            return psnr_per_image

    @staticmethod
    def get_ms_ssim(inp, otp):
        with tf.name_scope('mean_MS_SSIM'):
            return ms_ssim.MultiScaleSSIM(inp, otp, data_format='NCHW', name='MS-SSIM')


def _print_trainable_variables():
    print('*** tf.trainable_variables:')
    for v in tf.trainable_variables():
        print(v)

    print('*** TRAINABLE_RESOURCE_VARIABLES:')
    for v in tf.get_collection(tf.GraphKeys.TRAINABLE_RESOURCE_VARIABLES):
        print(v)


def _write_to_sheets(log_date, ae_config_rel_path, pc_config_rel_path,
                     description, git_ref,
                     log_dir_root, is_continue):
    try:
        with fasteners.InterProcessLock(sheets_logger.get_lock_file_p()):
            sheets_logger.insert_row(
                    log_date + ('c' if is_continue else ''),
                    os.environ.get('JOB_ID', 'N/A'),
                    ae_config_rel_path, pc_config_rel_path, description, '',
                    git_ref,
                    log_dir_root)
    except sheets_logger.GoogleSheetsAccessFailedException as e:
        print(e)


def _get_git_ref():
    """ :return HEAD commit as given by $QSUBA_GIT_REF """
    try:
        qsuba_git_ref = os.environ['QSUBA_GIT_REF']
        if 'tags' in qsuba_git_ref:
            return qsuba_git_ref
        git_commit = subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode()
        return '{} ({})'.format(qsuba_git_ref, git_commit[:16])
    except KeyError:
        return ''


def main():
    p = argparse.ArgumentParser()
    p.add_argument('autoencoder_config_path')
    p.add_argument('probclass_config_path')
    p.add_argument('--dataset_train', '-dtrain', default='imgnet_train', help=inputpipeline.get_dataset.__doc__)
    p.add_argument('--dataset_test', '-dtest', default='imgnet_test', help=inputpipeline.get_dataset.__doc__)
    p.add_argument('--dataset_codec_distance', '-dcodec', default='testset', help='See codec_distance.py')
    p.add_argument('--log_dir_root', '-o', default='logs', metavar='LOG_DIR_ROOT')
    p.add_argument('--log_interval_train', '-ltrain', type=int, default=100)
    p.add_argument('--log_interval_save', '-lsave', type=int, default=1000)
    p.add_argument('--log_interval_test', '-ltest', type=int, default=1000,
                   help='Set to -1 to skip testing, which saves memory.')
    p.add_argument('--log_run_metadata', '-lmeta', action='store_const', const=True)
    # TODO: rm
    p.add_argument('--summarize_gradients', '-lgrads', action='store_const', const=True)
    p.add_argument('--temporary', '-t', action='store_const', const=True, help='Append _TMP to LOG_DIR_ROOT')
    p.add_argument('--from_identity', metavar='IDENTITY_CKPT_DIR',
                   help='Like --restore IDENTITY_CKPT_DIR, but global_step and any variables matching *Adam* are not '
                        'restored and centers get sampled from the bottleneck with KMeans.')
    p.add_argument('--restore', '-r', metavar='RESTORE_DIR',
                   help='Path to ckpt dir to restore from.')
    p.add_argument('--restore_itr', '-i', type=int, default=-1,
                   help='Iteration to restore from. Use -1 for latest. Otherwise, restores the latest checkpoint '
                        'with iteration <= restore_itr')
    p.add_argument('--restore_continue', action='store_const', const=True,
                   help='If given, the log dir corresponding to the path given by RESTORE_DIR will be used to save '
                        'future logs and checkpoints.')
    p.add_argument('--restore_skip_vars', type=str,
                   help='Var names to skip, use comma to separate, e.g. "Adam, global_var".')
    p.add_argument('--ckpt_interval', type=float, default=1,
                   help='How often to keep checkpoints, in hours.')
    p.add_argument('--description', '-d', type=str,
                   help='Description, if given, is appended to Google Sheets')
    flags = p.parse_args()

    if flags.temporary:
        print('*** WARN: --temporary')
        time.sleep(1.5)
        flags.log_dir_root = flags.log_dir_root.rstrip(os.path.sep) + '_TMP'

    train_flags = TrainFlags(
            log_run_metadata=flags.log_run_metadata,
            log_interval_train=flags.log_interval_train,
            log_interval_test=flags.log_interval_test,
            log_interval_save=flags.log_interval_save,
            summarize_grads=flags.summarize_gradients)

    tf.set_random_seed(1234)
    train(autoencoder_config_path=flags.autoencoder_config_path,
          probclass_config_path=flags.probclass_config_path,
          restore_manager=RestoreManager.from_flags(flags),
          datasets=Datasets(flags.dataset_train, flags.dataset_test, flags.dataset_codec_distance),
          log_dir_root=flags.log_dir_root,
          train_flags=train_flags,
          ckpt_interval_hours=flags.ckpt_interval,
          description=flags.description if not flags.temporary else None)


if __name__ == '__main__':
    main()
