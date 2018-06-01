from os import path
import os
import glob
import shutil
import argparse

from saver import Saver
from fjcommon import no_op


def purge_checkpoints(log_dir_root, target_dir, verbose):
    vprint = print if verbose else no_op.NoOp
    ckpt_dir_glob = Saver.ckpt_dir_for_log_dir(path.join(log_dir_root, '*'))
    ckpt_dir_matches = sorted(glob.glob(ckpt_dir_glob))
    for ckpt_dir in ckpt_dir_matches:
        log_dir = Saver.log_dir_from_ckpt_dir(ckpt_dir)
        all_ckpts = Saver.all_ckpts_with_iterations(ckpt_dir)
        if len(all_ckpts) <= 5:
            vprint('Skipping {}'.format(log_dir))
            continue
        target_log_dir = path.join(target_dir, path.basename(log_dir))
        target_ckpt_dir = Saver.ckpt_dir_for_log_dir(target_log_dir)
        os.makedirs(target_ckpt_dir, exist_ok=True)
        ckpts_to_keep = {all_ckpts[2], all_ckpts[len(all_ckpts) // 2], all_ckpts[-1]}
        ckpts_to_move = set(all_ckpts) - ckpts_to_keep
        vprint('Moving to {}:'.format(target_ckpt_dir))
        for _, ckpt_to_move in ckpts_to_move:
            # ckpt_to_move is /path/to/dir/ckpt-7000, add a * to match ckpt-7000.data, .meta, .index
            for ckpt_file in glob.glob(ckpt_to_move + '*'):
                vprint('- {}'.format(ckpt_file))
                shutil.move(ckpt_file, target_ckpt_dir)


def main():
    p = argparse.ArgumentParser(usage='Delete all checkpoints except second one, last one and middle one.')
    p.add_argument('root_log_dir')
    p.add_argument('target_dir')
    p.add_argument('--verbose', '-v', action='store_const', const=True)

    flags = p.parse_args()
    purge_checkpoints(flags.root_log_dir, flags.target_dir, flags.verbose)

#
if __name__ == '__main__':
    main()



