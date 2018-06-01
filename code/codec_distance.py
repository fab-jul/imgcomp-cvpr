""" 
Calculates distance of some (bpp, metric) point (for some metric) to some codec on some dataset.
"""

import other_codecs
import os
import numpy as np
import scipy.interpolate
import constants
import logdir_helpers
from collections import defaultdict
from fjcommon import functools_ext as ft
import val_files

# how much of a bin must be filled
_REQUIRED_BINS = 0.99


DEFAULT_BPP_GRID = np.linspace(0.1, 1.4, 50)


# All these will be rendered into the plot, labelled according to their path using get_label
# Expected to be in constants.OTHER_CODECS_ROOT
# Create with other_codecs.py $IMG_DIR $OTHER_CODECS_ROOT/out_dir $MODE
CODECS = {
    'u100': {'jp2k': 'out_jp2k_Urban100_HR_crop',
             'bpg':  'out_bpg_Urban100_HR_crop',
             'jp':   'out_jp_Urban100_HR_crop'},
    'b100': {'jp2k': 'out_jp2k_B100_cropped',
             'bpg':  'out_bpg_B100_cropped',
             'jp':   'out_jp_B100_cropped'},
    'rf100': {'jp2k': 'out_jp2k_rf100',
              'bpg':  'out_bpg_rf100',
              'jp':   'out_jp_rf100_v3'},
    'testset': {'bpg': 'out_bpg_imagenet_256_train_val_128x128__100',
                'jp':  'out_jp_imagenet_256_train_val_128x128__100'},
    'kodak': {'bpg':  'out_bpg_kodak_v2',
              'jp2k': 'out_jp2k_Kodak',
              'jp':   'out_jp_Kodak',
              'webp': 'out_webp_kodak'},
    'cityscapes':  {'bpg': 'out_bpg_cityscapes'}
}



class CodecDistanceReadException(Exception):
    pass


class CodecDistance(object):
    def __init__(self, dataset, codec, metric):
        assert metric in other_codecs.SUPPORTED_METRICS, '{} not in {}'.format(metric, other_codecs.SUPPORTED_METRICS)
        if dataset not in CODECS.keys():
            raise CodecDistanceReadException('Dataset {} not in {}'.format(dataset, CODECS.keys()))
        if codec not in CODECS[dataset].keys():
            raise CodecDistanceReadException('Codec {} not in {}'.format(codec, CODECS[dataset].keys()))
        codec_dir = os.path.join(constants.OTHER_CODECS_ROOT, CODECS[dataset][codec])
        try:
            bpps, values = get_interpolated_values_bpg_jp2k(codec_dir, DEFAULT_BPP_GRID, metric)
        except ValueError as e:
            raise CodecDistanceReadException('Failed: {}'.format(e))
        self.f_bpp_meta = scipy.interpolate.interp1d(bpps, values, 'linear')

    def distance(self, bpp, value):
        codec_value = self.f_bpp_meta(bpp)  # may raise ValueError
        d = value - codec_value  # > 0 if we are better
        return d


def interpolator(measures_per_image_iter, grid, interp_mode='linear'):
    accumulated_values = np.zeros_like(grid, np.float64)
    # Count values per bin
    N = np.zeros_like(grid, np.int64)
    num_imgs = 0
    num_errors = 0
    for img_description, (bpps, values) in measures_per_image_iter:
        assert len(bpps) >= 2, 'Missing values for {}'.format(img_description)
        assert bpps[0] >= bpps[-1]

        num_imgs += 1
        # interpolation function
        try:
            fq = scipy.interpolate.interp1d(bpps, values, interp_mode)
        except ValueError as e:
            print(bpps, values)
            print(e)
            exit(1)
        for i, bpp in enumerate(grid):
            try:
                accumulated_values[i] += fq(bpp)
                N[i] += 1
            except ValueError as e:
                num_errors += 1
                continue
    try:
        grid, values = ft.unzip((bpp, m/n) for bpp, m, n in zip(grid, accumulated_values, N)
                                if n > _REQUIRED_BINS*num_imgs)
    except ValueError as e:
        raise e
    return grid, values


def get_interpolated_values_bpg_jp2k(bpg_or_jp2k_dir, grid, metric):
    """ :returns grid, values"""
    ps = other_codecs.all_measures_file_ps(bpg_or_jp2k_dir)
    if len(ps) == 0:
        raise CodecDistanceReadException('No matches in {}'.format(bpg_or_jp2k_dir))
    measures_per_image_iter = ((p, ft.unzip(sorted(other_codecs.read_measures(p, metric), reverse=True))) for p in ps)
    return interpolator(measures_per_image_iter, grid, interp_mode='linear')


def get_measures_readers(log_dir_root, job_ids, dataset):
    if job_ids == 'NA':  # TODO
        return []
    missing = []
    measures_readers = []

    for job_id, ckpt_dir in zip(job_ids.split(','), logdir_helpers.iter_ckpt_dirs(log_dir_root, job_ids)):
        val_dirs = val_files.ValidationDirs(ckpt_dir, log_dir_root, dataset)
        try:
            measures_reader = val_files.MeasuresReader(val_dirs.out_dir)
            measures_readers.append(measures_reader)
        except FileNotFoundError:
            missing.append(job_id)
    if missing:
        print('Missing measures files for:\n{}'.format(','.join(missing)))
    # uniquify
    m = [val_files.MeasuresReader(o) for o in {m.out_dir for m in measures_readers}]
    return m


def interpolate_ours(measures_readers, grid, interp_mode, metric):
    measures_per_image = defaultdict(list)
    for measures_reader in measures_readers:
        for img_name, bpp, value in measures_reader.iter_metric(metric):
            measures_per_image[img_name].append((bpp, value))

    # Make sure every job has a value for every image
    for img_name, values in measures_per_image.items():
        assert len(values) == len(measures_readers), '{}: {}'.format(img_name, len(values))

    return interpolator(
            ((img_name, ft.unzip(sorted(bpps_values, reverse=True)))
             for img_name, bpps_values in measures_per_image.items()),
            grid, interp_mode)


