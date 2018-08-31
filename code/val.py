import tensorflow as tf

import numpy as np
import argparse
import logdir_helpers
import val_images
import constants
import autoencoder
import probclass
import glob
import bits
import imageio
import ms_ssim_np
from codec_distance import CodecDistance, CodecDistanceReadException
from fjcommon import tf_helpers, config_parser

from images_iterator import ImagesIterator
from saver import Saver
import os
from os import path
import skimage.measure
from collections import defaultdict
from collections import namedtuple

from val_files import ValidationDirs, MeasuresWriter

import bpp_helpers


_VALIDATION_INFO_STR = """
- VALIDATION -------------------------------------------------------------------"""


_CKPT_ITR_INFO_STR = """- Validating ckpt {} ----------"""


OutputFlags = namedtuple('OutputFlags', ['save_ours', 'ckpt_step', 'real_bpp'])


def validate(val_dirs: ValidationDirs, images_iterator: ImagesIterator, flags: OutputFlags):
    """
    Saves in val_dirs.log_dir/val/dataset_name/measures.csv:
        - `img_name,bpp,psnr,ms-ssim forall img_name`
    """
    print(_VALIDATION_INFO_STR)

    validated_checkpoints = val_dirs.get_validated_checkpoints()  # :: [10000, 18000, ..., 256000], ie, [int]
    all_ckpts = Saver.all_ckpts_with_iterations(val_dirs.ckpt_dir)
    if len(all_ckpts) == 0:
        print('No checkpoints found in {}'.format(val_dirs.ckpt_dir))
        return
    # if ckpt_step is -1, then all_ckpt[:-1:flags.ckpt_step] === [] because of how strides work
    ckpt_to_check = all_ckpts[:-1:flags.ckpt_step] + [all_ckpts[-1]]  # every ckpt_step-th checkpoint plus the last one
    if flags.ckpt_step == -1:
        assert len(ckpt_to_check) == 1
    print('Validating {}/{} checkpoints (--ckpt_step {})...'.format(
            len(ckpt_to_check), len(all_ckpts), flags.ckpt_step))

    missing_checkpoints = [(ckpt_itr, ckpt_path)
                           for ckpt_itr, ckpt_path in ckpt_to_check
                           if ckpt_itr not in validated_checkpoints]
    if len(missing_checkpoints) == 0:
        print('All checkpoints validated, stopping...')
        return

    # ---

    # create networks
    autoencoder_config_path, probclass_config_path = logdir_helpers.config_paths_from_log_dir(
            val_dirs.log_dir, base_dirs=[constants.CONFIG_BASE_AE, constants.CONFIG_BASE_PC])
    ae_config, ae_config_rel_path = config_parser.parse(autoencoder_config_path)
    pc_config, pc_config_rel_path = config_parser.parse(probclass_config_path)

    ae_cls = autoencoder.get_network_cls(ae_config)
    pc_cls = probclass.get_network_cls(pc_config)

    # Instantiate autoencoder and probability classifier
    ae = ae_cls(ae_config)
    pc = pc_cls(pc_config, num_centers=ae_config.num_centers)

    x_val_ph = tf.placeholder(tf.uint8, (3, None, None), name='x_val_ph')
    x_val_uint8 = tf.expand_dims(x_val_ph, 0, name='batch')
    x_val = tf.to_float(x_val_uint8, name='x_val')

    enc_out_val = ae.encode(x_val, is_training=False)
    x_out_val = ae.decode(enc_out_val.qhard, is_training=False)

    bc_val = pc.bitcost(enc_out_val.qbar, enc_out_val.symbols, is_training=False, pad_value=pc.auto_pad_value(ae))
    bpp_val = bits.bitcost_to_bpp(bc_val, x_val)

    x_out_val_uint8 = tf.cast(x_out_val, tf.uint8, name='x_out_val_uint8')
    # Using numpy implementation due to dynamic shapes
    msssim_val = ms_ssim_np.tf_msssim_np(x_val_uint8, x_out_val_uint8, data_format='NCHW')
    psnr_val = psnr_np(x_val_uint8, x_out_val_uint8)

    restorer = Saver(val_dirs.ckpt_dir, var_list=Saver.get_var_list_of_ckpt_dir(val_dirs.ckpt_dir))

    # create fetch_dict
    fetch_dict = {
        'bpp': bpp_val,
        'ms-ssim': msssim_val,
        'psnr': psnr_val,
    }

    if flags.real_bpp:
        fetch_dict['sym'] = enc_out_val.symbols  # NCHW

    if flags.save_ours:
        fetch_dict['img_out'] = x_out_val_uint8

    # ---
    fw = tf.summary.FileWriter(val_dirs.out_dir, graph=tf.get_default_graph())

    def full_summary_tag(summary_name):
        return '/'.join(['val', images_iterator.dataset_name, summary_name])

    # Distance
    try:
        codec_distance_ms_ssim = CodecDistance(images_iterator.dataset_name, codec='bpg', metric='ms-ssim')
        codec_distance_psnr = CodecDistance(images_iterator.dataset_name, codec='bpg', metric='psnr')
    except CodecDistanceReadException as e:  # no codec distance values stored for the current setup
        print('*** Distance to BPG not available for {}:\n{}'.format(images_iterator.dataset_name, str(e)))
        codec_distance_ms_ssim = None
        codec_distance_psnr = None

    # Note that for each checkpoint, the structure of the network will be the same. Thus the pad depending image
    # loading can be cached.

    # create session
    with tf_helpers.create_session() as sess:
        if flags.real_bpp:
            pred = probclass.PredictionNetwork(pc, pc_config, ae.get_centers_variable(), sess)
            checker = probclass.ProbclassNetworkTesting(pc, ae, sess)
            bpp_fetcher = bpp_helpers.BppFetcher(pred, checker)

        fetcher = sess.make_callable(fetch_dict, feed_list=[x_val_ph])

        last_ckpt_itr = missing_checkpoints[-1][0]
        for ckpt_itr, ckpt_path in missing_checkpoints:
            if not ckpt_still_exists(ckpt_path):
                # May happen if job is still training
                print('Checkpoint disappeared: {}'.format(ckpt_path))
                continue

            print(_CKPT_ITR_INFO_STR.format(ckpt_itr))

            restorer.restore_ckpt(sess, ckpt_path)

            values_aggregator = ValuesAggregator('bpp', 'ms-ssim', 'psnr')

            # truncates the previous measures.csv file! This way, only the last valid checkpoint is saved.
            measures_writer = MeasuresWriter(val_dirs.out_dir)

            # ----------------------------------------
            # iterate over images
            # images are padded to work with current auto encoder
            for img_i, (img_name, img_content) in enumerate(images_iterator.iter_imgs(pad=ae.get_subsampling_factor())):
                otp = fetcher(img_content)
                measures_writer.append(img_name, otp)

                if flags.real_bpp:
                    # Calculate
                    bpp_real, bpp_theory = bpp_fetcher.get_bpp(
                            otp['sym'], bpp_helpers.num_pixels_in_image(img_content))

                    # Logging
                    bpp_loss = otp['bpp']
                    diff_percent_tr = (bpp_theory/bpp_real) * 100
                    diff_percent_lt = (bpp_loss/bpp_theory) * 100
                    print('BPP: Real         {:.5f}\n'
                          '     Theoretical: {:.5f} [{:5.1f}% of real]\n'
                          '     Loss:        {:.5f} [{:5.1f}% of real]'.format(
                            bpp_real, bpp_theory, diff_percent_tr, bpp_loss, diff_percent_lt))
                    assert abs(bpp_theory - bpp_loss) < 1e-3, 'Expected bpp_theory to match loss! Got {} and {}'.format(
                            bpp_theory, bpp_loss)

                if flags.save_ours and ckpt_itr == last_ckpt_itr:
                    save_img(img_name, otp['img_out'], val_dirs)

                values_aggregator.update(otp)

                print('{: 10d} {img_name} | Mean: {avgs}'.format(
                        img_i, img_name=img_name, avgs=values_aggregator.averages_str()),
                      end=('\r' if not flags.real_bpp else '\n'), flush=True)

            measures_writer.close()

            print()  # add newline
            avgs = values_aggregator.averages()
            avg_bpp, avg_ms_ssim, avg_psnr = avgs['bpp'], avgs['ms-ssim'], avgs['psnr']

            tf_helpers.log_values(fw,
                                  [(full_summary_tag('avg_bpp'), avg_bpp),
                                   (full_summary_tag('avg_ms_ssim'), avg_ms_ssim),
                                   (full_summary_tag('avg_psnr'), avg_psnr)],
                                  iteration=ckpt_itr)

            if codec_distance_ms_ssim and codec_distance_psnr:
                try:
                    d_ms_ssim = codec_distance_ms_ssim.distance(avg_bpp, avg_ms_ssim)
                    d_pnsr = codec_distance_psnr.distance(avg_bpp, avg_psnr)
                    print('Distance to BPG: {:.3f} ms-ssim // {:.3f} psnr'.format(d_ms_ssim, d_pnsr))
                    tf_helpers.log_values(fw,
                                          [(full_summary_tag('distance_BPG_MS-SSIM'), d_ms_ssim),
                                           (full_summary_tag('distance_BPG_PSNR'),    d_pnsr)],
                                          iteration=ckpt_itr)
                except ValueError as e:  # out of range errors from distance calls
                    print(e)

            val_dirs.add_validated_checkpoint(ckpt_itr)

    print('Validation completed {}'.format(val_dirs))


def save_img(img_name, img_out, val_dirs):
    assert img_name.endswith('.png')
    assert img_out.ndim == 4 and img_out.shape[1] == 3, 'Expected NCHW, got {}'.format(img_out)

    img_dir = path.join(val_dirs.out_dir, 'imgs')
    os.makedirs(img_dir, exist_ok=True)
    img_out = np.transpose(img_out[0, :, :, :], (1, 2, 0))  # Make HWC
    img_out_p = path.join(img_dir, img_name)
    print('Saving {}...'.format(img_out_p))
    imageio.imsave(img_out_p, img_out)


def psnr_np(img1, img2):
    assert tf.uint8.is_compatible_with(img1.dtype), 'Expected uint8 intput'
    assert tf.uint8.is_compatible_with(img2.dtype), 'Expected uint8 intput'

    def _psnr(_img1, _img2):
        return np.float32(skimage.measure.compare_psnr(_img1, _img2))

    with tf.name_scope('psnr_np'):
        v = tf.py_func(_psnr, [img1, img2], tf.float32, stateful=False, name='PSNR')
        v.set_shape(())
        return v


class ValuesAggregator(object):
    def __init__(self, *tags_to_agregate):
        self._tags_to_values = defaultdict(list)  # log tag -> [log value]
        self.tags_to_agregate = tags_to_agregate

    def update(self, fetch_dict_out):
        for tag, value in fetch_dict_out.items():
            if tag in self.tags_to_agregate:
                assert not np.isnan(value), 'nan encountered in {}'.format(fetch_dict_out)
                self._tags_to_values[tag].append(value)

    def averages(self):
        return {tag: np.mean(values) for tag, values in self._tags_to_values.items()}

    def averages_str(self, joiner=', '):
        mean_values = self.averages()
        avergaes_sorted = tuple((tag, mean_values[tag]) for tag in self.tags_to_agregate)  # sort by tags_to_agregate
        return joiner.join('{}: {:.3f}'.format(tag, value) for tag, value in avergaes_sorted)


def ckpt_still_exists(ckpt_path):
    ckpt_files = glob.glob(ckpt_path + '*')
    return len(ckpt_files) > 0


def main():
    p = argparse.ArgumentParser()
    p.add_argument('log_dir_root', help='Path to dir containing log_dirs.')
    p.add_argument('job_ids', help='Comma separated list of job_ids.')
    p.add_argument('images')
    p.add_argument('--save_ours', '-o', action='store_const', const=True,
                   help='If given, store output images in VAL_OUT/imgs.')
    p.add_argument('--how_many', type=int, help='Number of images to output')
    p.add_argument('--image_cache_max', '-cache', type=int, default=500, help='Cache max in [MB]. Set to 0 to disable.')
    p.add_argument('--restore_itr', '-i', type=int)
    p.add_argument('--ckpt_step', '-s', type=int, default=2,
                   help='Every CKPT_STEP-th checkpoint will be validated. Set to 1 to validate all of them. '
                        'Last checkpoint will always be validated. Set to -1 to only validate last.')
    p.add_argument('--reset', action='store_const', const=True, help='Remove previous output')
    p.add_argument('--real_bpp', action='store_const', const=True,
                   help='If given, calculate real bpp using arithmetic encoding. Note: in our experiments, '
                        'this matches the theoretical bpp up to 1% precision. Note: this is very slow.')

    flags, unknown_flags = p.parse_known_args()

    if unknown_flags:
        print('Unknown flags: {}'.format(unknown_flags))

    image_paths, dataset_name = val_images.get_image_paths(flags.images)
    images_iterator = ImagesIterator(image_paths[:flags.how_many], dataset_name, flags.image_cache_max)
    val_flags = OutputFlags(flags.save_ours, flags.ckpt_step, flags.real_bpp)

    for ckpt_dir in logdir_helpers.iter_ckpt_dirs(flags.log_dir_root, flags.job_ids):
        try:
            validate(ValidationDirs(ckpt_dir, flags.log_dir_root, dataset_name, flags.reset),
                     images_iterator,
                     val_flags)
        except tf.errors.NotFoundError as e:
            # happens if ckpt was deleted while validation
            print('*** Caught {}'.format(e))
            continue
        tf.reset_default_graph()
    print('*** All given job_ids validated.')


if __name__ == '__main__':
    main()


