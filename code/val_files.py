import pickle
import shutil
from os import path
import time

import logdir_helpers
from saver import Saver


_MEASURES_FILE_NAME = 'measures.csv'


class ValidationDirs(object):
    """
    Keeps track of which iterations have already been validated in the file
        log_dir/{log_date} {dataset_name}/validated_ckpts.pkl
    which containes a pickled python list of ints.

    """
    def __init__(self, ckpt_dir, log_dir_root, dataset_name, reset=False):
        self.ckpt_dir = ckpt_dir
        self.log_dir = Saver.log_dir_from_ckpt_dir(self.ckpt_dir)
        self.log_dir_root = log_dir_root
        self.dataset_name = dataset_name

        log_date = logdir_helpers.log_date_from_log_dir(self.log_dir)
        self.out_dir = path.join(self.log_dir_root, '{log_date} {dataset_name}'.format(
                log_date=log_date, dataset_name=dataset_name))
        self.validated_ckpts_f = path.join(self.out_dir, 'validated_ckpts.pkl')

        if reset:
            self._reset()

    @staticmethod
    def job_id_from_out_dir(out_dir):
        base = path.basename(out_dir)  # should be {log_date} {dataset_name}
        return logdir_helpers.log_date_from_log_dir(base)  # may raise ValueError

    def _reset(self):
        if path.isdir(self.out_dir):
            print('*** rm -rf {}'.format(self.out_dir))
            time.sleep(1.5)
            shutil.rmtree(self.out_dir)

    def get_validated_checkpoints(self):
        if not path.exists(self.validated_ckpts_f):
            return []
        with open(self.validated_ckpts_f, 'rb') as f:
            return pickle.load(f)

    def add_validated_checkpoint(self, ckpt_itr):
        validated_checkpoints = self.get_validated_checkpoints()
        validated_checkpoints.append(ckpt_itr)
        with open(self.validated_ckpts_f, 'wb') as f:
            pickle.dump(validated_checkpoints, f)

    def __str__(self):
        return 'Validation out dir: {}, validated: {}'.format(
                self.out_dir, ' '.join(map(str, self.get_validated_checkpoints())))


class MeasuresWriter(object):
    def __init__(self, out_dir):
        p = path.join(out_dir, _MEASURES_FILE_NAME)
        self.fout = open(p, 'w')
        self.fout.write('img_name,bpp,ms-ssim,psnr\n')

    def append(self, img_name, otp):
        """
        :param img_name:
        :param otp: dict containing values for keys 'bpp', 'ms-ssim', 'psnr'
        :return:
        """
        self.fout.write('{},{},{},{}\n'.format(img_name, otp['bpp'], otp['ms-ssim'], otp['psnr']))

    def close(self):
        self.fout.close()


class MeasuresReader(object):
    def __init__(self, out_dir):
        self.out_dir = out_dir
        self.p = path.join(out_dir, _MEASURES_FILE_NAME)
        if not path.isfile(self.p):
            raise FileNotFoundError('No {} for {}'.format(_MEASURES_FILE_NAME, out_dir))

    def iter_metric(self, metric):
        with open(self.p, 'r') as f:
            fit = iter(f)
            next(fit)  # skip header
            for l in fit:
                img_name, bpp, ms_ssim, psnr = l.split(',')
                try:
                    value = {'ms-ssim': ms_ssim, 'psnr': psnr}[metric]
                    yield img_name, float(bpp), float(value)
                except KeyError:
                    raise ValueError('Invalid metric: {}'.format(metric))

    # for convenience
    def get_job_id(self):
        return ValidationDirs.job_id_from_out_dir(self.out_dir)

