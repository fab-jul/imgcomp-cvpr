import os
import argparse

import numpy as np
from fjcommon import functools_ext as ft
from fjcommon.iterable_ext import flag_first_iter

import matplotlib as mpl
mpl.rcParams['text.latex.unicode'] = True
mpl.use('Agg')  # No display
from matplotlib import pyplot as plt

import constants
from codec_distance import get_interpolated_values_bpg_jp2k, get_measures_readers, interpolate_ours, \
    DEFAULT_BPP_GRID, CODECS


LABEL_OURS = 'Ours'
LABEL_RB = 'Rippel \& Bourdev'
LABEL_BPG = 'BPG'
LABEL_JP2K = 'JPEG2000'
LABEL_JP = 'JPEG'
LABEL_WEBP = 'WebP'
LABEL_THEIS = 'Theis et al.'
LABEL_JOHNSTON = 'Johnston et al.'
LABEL_BALLE = 'Ballé et al.'
TITLES = {'u100': 'Urban100',
          'b100': 'B100',
          'rf100': 'ImageNetVal',
          'kodak': 'Kodak',
          'testset': 'TestSet'}


def get_label_from_codec_short_name(codec_short_name):
    return {'bpg': LABEL_BPG,
            'jp2k': LABEL_JP2K,
            'jp': LABEL_JP,
            'webp': LABEL_WEBP}[codec_short_name]


CVPR_FIG1 = [
    (0.1265306, 0.9289356),
    (0.1530612, 0.9417454),
    (0.1795918, 0.9497924),
    (0.2061224, 0.9553684),
    (0.2326531, 0.9598574),
    (0.2591837, 0.9636625),
    (0.2857143, 0.9668663),
    (0.3122449, 0.9695684),
    (0.3387755, 0.9718446),
    (0.3653061, 0.9738012),
    (0.3918367, 0.9755308),
    (0.4183673, 0.9770696),
    (0.4448980, 0.9784622),
    (0.4714286, 0.9797252),
    (0.4979592, 0.9808753),
    (0.5244898, 0.9819255),
    (0.5510204, 0.9828875),
    (0.5775510, 0.9837722),
    (0.6040816, 0.9845877),
    (0.6306122, 0.9853407),
    (0.6571429, 0.9860362),
    (0.6836735, 0.9866768),
    (0.7102041, 0.9872690),
    (0.7367347, 0.9878184),
    (0.7632653, 0.9883268),
    (0.7897959, 0.9887977),
    (0.8163265, 0.9892346),
    (0.8428571, 0.9896379)]


# Transcribed from paper
_RIPPEL_KODAK = [
    (.095, .92),
    (.14,  .94),
    (.2,   .956),
    (.3,   .97),
    (.4,   .9783),
    (.5,   .983),
    (.6,   .9858),
    (.7,   .9880),
    (.8,   .9897),
    (.9,   .9914),
    (1.0,  .9923),
    (1.1,  .9935),
    (1.2,  .994),
    (1.3,  .9946),
    (1.4,  .9954)
]


def plot_ours_mean(measures_readers, metric, color, show_ids):
    if not show_ids:
        show_ids = []
    ops = []
    for first, measures_reader in flag_first_iter(measures_readers):
        this_op_bpps = []
        this_op_values = []
        for img_name, bpp, value in measures_reader.iter_metric(metric):
            this_op_bpps.append(bpp)
            this_op_values.append(value)
        ours_mean_bpp, ours_mean_value = np.mean(this_op_bpps), np.mean(this_op_values)
        ops.append((ours_mean_bpp, ours_mean_value))
        plt.scatter(ours_mean_bpp, ours_mean_value, marker='x', zorder=10, color=color,
                    label='Ours' if first else None)
    for (bpp, value), job_id in zip(sorted(ops), show_ids):
        plt.annotate(job_id, (bpp + 0.04, value),
                     horizontalalignment='bottom', verticalalignment='center')


def interpolated_curve(log_dir_root, job_ids, dataset,
                       grid, interp_mode,
                       plot_interp_of_ours, plot_mean_of_ours, plot_ids_of_ours,
                       metric,
                       x_range, y_range, use_latex,
                       output_path, paper_plot):
    if not output_path:
        output_path = 'plot_{}.png'.format(TITLES[dataset])

    cmap = plt.cm.get_cmap('cool')

    style = {
        LABEL_OURS: ('0', '-', 3),
        LABEL_RB: (cmap(0.9), '-', 1.5),
        LABEL_BPG: (cmap(0.7), '-', 1.5),
        LABEL_JP2K: (cmap(0.45), '-', 1.5),
        LABEL_JP: (cmap(0.2), '-', 1.5),
        LABEL_WEBP: (cmap(0.1), '-', 1.5),
        LABEL_JOHNSTON: (cmap(0.7), '--', 1.5),
        LABEL_BALLE: (cmap(0.45), '--', 1.5),
        LABEL_THEIS: (cmap(0.2), '--', 1.5),
    }

    pos = {
        LABEL_OURS: 10,
        LABEL_RB: 9,
        LABEL_JOHNSTON: 8,
        LABEL_BPG: 7,
        LABEL_BALLE: 6,
        LABEL_JP2K: 5,
        LABEL_THEIS: 4,
        LABEL_JP: 3,
        LABEL_WEBP: 2,
        'Fig. 1': 11
    }

    plt.figure(figsize=(6, 6))
    if use_latex:
        plt.rc('text', usetex=True)
        plt.rc('font', family='serif', serif=['Computer Modern Roman'])


    for codec_short_name, measures_dir in CODECS[dataset].items():
        measures_dir = os.path.join(constants.OTHER_CODECS_ROOT, measures_dir)
        label = get_label_from_codec_short_name(codec_short_name)
        col, line_style, line_width = style[label]
        assert os.path.exists(measures_dir), measures_dir
        this_grid, this_msssims = get_interpolated_values_bpg_jp2k(measures_dir, grid, metric)
        dashes = (5,1) if line_style == '--' else []
        plt.plot(this_grid, this_msssims, label=label, linewidth=line_width, color=col, dashes=dashes)

    if dataset == 'kodak':
        for name, data in [(LABEL_RB, _RIPPEL_KODAK)]:
            print('hi')
            col, line_style, line_width = style[name]
            dashes = (5, 1) if line_style == '--' else []
            plt.plot(*ft.unzip(data), label=name, color=col, linewidth=line_width, dashes=dashes)

    for job_ids in job_ids.split(';'):  
        measures_readers = get_measures_readers(log_dir_root, job_ids, dataset)
        print('\n'.join(m.p for m in measures_readers))

        if measures_readers:  # may be empty if no job_ids are passed
            col, line_style, line_width = style['Ours']
            if plot_interp_of_ours:
                ours_grid, ours_msssim = interpolate_ours(measures_readers, grid, interp_mode, metric)
                dashes = (5, 1) if line_style == '--' else []
                plt.plot(ours_grid, ours_msssim, label='Ours', color=col, linewidth=line_width, dashes=dashes)
            if plot_mean_of_ours:
                plot_ours_mean(measures_readers, metric, col, plot_ids_of_ours)

    if paper_plot:
        col, line_style, line_width = style['Ours']
        dashes = (5, 1) if line_style == '--' else []
        plt.plot(*ft.unzip(CVPR_FIG1), label='Fig. 1', color=col, linewidth=line_width, dashes=dashes)


    plt.title('{} on {}'.format(metric.upper(), TITLES[dataset]))
    plt.xlabel('bpp', labelpad=-5)
    plt.grid()

    ax = plt.gca()
    handles, labels = ax.get_legend_handles_labels()
    labels, handles = zip(*sorted(zip(labels, handles), reverse=True, key=lambda t: pos[t[0]]))
    ax.legend(handles, labels, loc=4, prop={'size': 12}, fancybox=True, framealpha=0.7)

    ax.yaxis.grid(b=True, which='both', color='0.8', linestyle='-')
    ax.xaxis.grid(b=True, which='major', color='0.8', linestyle='-')
    ax.set_axisbelow(True)

    ax.minorticks_on()
    ax.yaxis.set_minor_locator(mpl.ticker.AutoMinorLocator(2))

    plt.xlim(x_range)
    plt.ylim(y_range)
    print('Saving {}...'.format(output_path))
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('log_dir_root', help='Path to dir containing log_dirs.')
    p.add_argument('job_ids', help='Comma separated list of job_ids.')
    p.add_argument('images')
    p.add_argument('--x_range', default='0,1.2')
    p.add_argument('--y_range', default='0.85,1.0')
    p.add_argument('--latex', action='store_true')
    p.add_argument('--output_path', '-o', help='Path to store plot. Defaults to plot_DATASET.png.')
    p.add_argument('--style', nargs='+', default=['interp'], choices=['interp', 'mean'])
    p.add_argument('--paper_plot', action='store_true')
    p.add_argument('--ids', help='If given with --style mean, label mean points with these ids.',
                   nargs='+')
    flags = p.parse_args()

    range_to_floats = lambda r: tuple(map(float, r.split(',')))
    interpolated_curve(flags.log_dir_root, flags.job_ids, flags.images,
                       DEFAULT_BPP_GRID, 'quadratic',
                       plot_interp_of_ours='interp' in flags.style,
                       plot_mean_of_ours='mean' in flags.style,
                       plot_ids_of_ours=flags.ids,
                       metric='ms-ssim',
                       x_range=range_to_floats(flags.x_range), y_range=range_to_floats(flags.y_range),
                       use_latex=flags.latex,
                       output_path=flags.output_path,
                       paper_plot=flags.paper_plot)


if __name__ == '__main__':
    main()

