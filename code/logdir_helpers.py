import glob
from datetime import datetime, timedelta

import fasteners
import re
import os
from os import path

from saver import Saver


_LOG_DATE_FORMAT = "%m%d_%H%M"
_RESTORE_PREFIX = 'RESTORE@'


def iter_ckpt_dirs(log_dir_root, job_ids_str):
    assert os.path.exists(log_dir_root), 'Invalid log dir: {}'.format(log_dir_root)
    job_ids = job_ids_str.strip().replace(';', ',').split(',')
    assert len(job_ids) > 0, 'No job_ids!'
    for job_id in job_ids:
        # ckpt_dir_for_log_dir appends 'ckpts', which ensures that we only get training log dirs as matches,
        # and not other or previous validation dir.
        ckpt_dir_glob = Saver.ckpt_dir_for_log_dir(path.join(log_dir_root, job_id + '*'))
        ckpt_dir_matches = glob.glob(ckpt_dir_glob)
        if len(ckpt_dir_matches) == 0:
            print('*** ERR: No matches for {}'.format(ckpt_dir_glob))
            continue
        if len(ckpt_dir_matches) > 1:
            print('*** ERR: Multiple matches for {}: {}'.format(ckpt_dir_glob, '\n'.join(ckpt_dir_matches)))
            continue
        yield ckpt_dir_matches[0]


def create_unique_log_dir(config_rel_paths, log_dir_root, line_breaking_chars_pat=r'[-]', restore_dir=None):
    """
    0117_1704 repr@soa3_med_8e*5_deePer_b50_noHM_C16 repr@v2_res_shallow RESTORE@path@to@restore@0115_1340
    :param config_rel_paths:
    :param log_dir_root:
    :param line_breaking_chars_pat:
    :return:
    """
    if any(':' in config_rel_path for config_rel_path in config_rel_paths):
        raise ValueError('":" not allowed in paths, got {}'.format(config_rel_paths))

    def prep_path(p):
        p = p.replace(path.sep, '@')
        return re.sub(line_breaking_chars_pat, '*', p)

    postfix_dir_name = ' '.join(map(prep_path, config_rel_paths))
    if restore_dir:
        restore_dir_root, restore_job_component = _split_log_dir(restore_dir)
        restore_dir_root = restore_dir_root.replace(path.sep, '@')
        restore_job_id = log_date_from_log_dir(restore_job_component)
        postfix_dir_name += ' {restore_prefix}{root}@{job_id}'.format(
                restore_prefix=_RESTORE_PREFIX, root=restore_dir_root, job_id=restore_job_id)
    return _mkdir_threadsafe_unique(log_dir_root, datetime.now(), postfix_dir_name)


def _split_log_dir(log_dir):
    """
    given
        some/path/to/job/dir/0101_1818 ae_config pc_config/ckpts
    or
        some/path/to/job/dir/0101_1818 ae_config pc_config
    returns
        tuple some/path/to/job/dir, 0101_1818 ae_config pc_config
    """
    log_dir_root = []
    job_component = None

    for comp in log_dir.split(path.sep):
        try:
            log_date_from_log_dir(comp)
            job_component = comp
            break  # this component is an actual log dir. stop and return components
        except ValueError:
            log_dir_root.append(comp)

    assert job_component is not None, 'Invalid log_dir: {}'.format(log_dir)
    return path.sep.join(log_dir_root), job_component


def _mkdir_threadsafe_unique(log_dir_root, log_date, postfix_dir_name):
    os.makedirs(log_dir_root, exist_ok=True)
    # Make sure only one process at a time writes into log_dir_root
    with fasteners.InterProcessLock(os.path.join(log_dir_root, 'lock')):
        return _mkdir_unique(log_dir_root, log_date, postfix_dir_name)


def _mkdir_unique(log_dir_root, log_date, postfix_dir_name):
    log_date_str = log_date.strftime(_LOG_DATE_FORMAT)
    if _log_dir_with_log_date_exists(log_dir_root, log_date):
        print('Log dir starting with {} exists...'.format(log_date_str))
        return _mkdir_unique(log_dir_root, log_date + timedelta(minutes=1), postfix_dir_name)

    log_dir = path.join(log_dir_root, '{log_date_str} {postfix_dir_name}'.format(
        log_date_str=log_date_str,
        postfix_dir_name=postfix_dir_name))
    os.makedirs(log_dir)
    return log_dir


def _log_dir_with_log_date_exists(log_dir_root, log_date):
    log_date_str = log_date.strftime(_LOG_DATE_FORMAT)
    all_log_dates = set()
    for log_dir in os.listdir(log_dir_root):
        try:
            all_log_dates.add(log_date_from_log_dir(log_dir))
        except ValueError:
            continue
    return log_date_str in all_log_dates


def log_date_from_log_dir(log_dir):
    # extract {log_date} from LOG_DIR/{log_date} {netconfig} {probconfig}
    possible_log_date = os.path.basename(log_dir).split(' ')[0]
    if not is_log_date(possible_log_date):
        raise ValueError('Invalid log dir: {}'.format(log_dir))
    return possible_log_date


def is_log_date(possible_log_date):
    try:
        datetime.strptime(possible_log_date, _LOG_DATE_FORMAT)
        return True
    except ValueError:
        return False


def config_paths_from_log_dir(log_dir, base_dirs):
    log_dir = path.basename(log_dir.strip(path.sep))

    # log_dir == {now} {netconfig} {probconfig} [RESTORE@some_dir@XXXX_YYYY], get [netconfig, probconfig]
    comps = log_dir.split(' ')
    assert is_log_date(comps[0]), 'Invalid log_dir: {}'.format(log_dir)
    comps = [c for c in comps[1:] if _RESTORE_PREFIX not in c]
    assert len(comps) <= len(base_dirs), 'Expected as many config components as base dirs: {}, {}'.format(
            comps, base_dirs)

    def get_real_path(base, prepped_p):
        p_glob = prepped_p.replace('@', path.sep)
        p_glob = path.join(base, p_glob)  # e.g., ae_configs/p_glob
        glob_matches = glob.glob(p_glob)
        # We always only replace one character with *, so filter for those.
        # I.e. lr1e-5 will become lr1e*5, which will match lr1e-5 but also lr1e-4.5
        glob_matches_of_same_len = [g for g in glob_matches if len(g) == len(p_glob)]
        if len(glob_matches_of_same_len) != 1:
            raise ValueError('Cannot find config on disk: {} (matches: {})'.format(p_glob, glob_matches_of_same_len))
        return glob_matches_of_same_len[0]

    return tuple(get_real_path(base_dir, comp) for base_dir, comp in zip(base_dirs, comps))
