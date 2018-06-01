import tensorflow as tf
from os import path
import os
import re
import pickle
from fjcommon import tf_helpers


_CKPT_DIR_NAME = 'ckpts'
_CKPT_FN = 'ckpt'


def save_vars(v_names, p):
    with open(p, 'w') as f:
        f.write('\n'.join(v_name for v_name in v_names))



class VarNames(object):
    def __init__(self, ckpt_dir):
        self._pickle_p = path.join(ckpt_dir, 'var_names.pkl')

    def exists(self):
        return path.exists(self._pickle_p)

    def read(self, skip_var_names=None):
        assert self.exists()
        if skip_var_names is None:
            skip_var_names = []
        with open(self._pickle_p, 'rb') as f:
            all_v = pickle.load(f)
            filtered_v = [v for v in all_v
                          if not any(skip in v for skip in skip_var_names)]
            num_skippped = len(all_v) - len(filtered_v)
            if num_skippped > 0:
                print('Skipping {} variables matching {}...'.format(num_skippped, '|'.join(skip_var_names)))
            return filtered_v

    def write(self, var_names):
        assert isinstance(var_names, list)
        # print('*** Saving Var names:\n{}\n***'.format('\n'.join(var_names)))
        with open(self._pickle_p, 'wb') as f:
            return pickle.dump(var_names, f)


class Saver(object):
    @staticmethod
    def is_ckpt_dir(p):
        return path.basename(p) == _CKPT_DIR_NAME

    @staticmethod
    def ckpt_dir_for_log_dir(log_dir):
        return path.join(log_dir, _CKPT_DIR_NAME)

    @staticmethod
    def log_dir_from_ckpt_dir(ckpt_dir):
        assert Saver.is_ckpt_dir(ckpt_dir)
        return path.dirname(ckpt_dir)

    def __init__(self, ckpt_dir, **kwargs_saver):
        """
        :param ckpt_dir: where to save data
        :param kwargs_saver: Passed on to the tf.train.Saver that will be created
        """
        os.makedirs(ckpt_dir, exist_ok=True)
        self.ckpt_dir = ckpt_dir
        self.ckpt_base_file_path = path.join(ckpt_dir, _CKPT_FN)

        all_saveable_vars = tf_helpers.all_saveable_objects()
        var_list = kwargs_saver.get('var_list', all_saveable_vars)
        var_names = VarNames(ckpt_dir)
        if not var_names.exists():
            print('Saver for {} saves {} variables...'.format(self.ckpt_dir, len(var_list)))
            var_names.write([v.name for v in var_list])

        unrestored_vars = [v for v in all_saveable_vars if v not in var_list]
        if unrestored_vars:
            print('Found {} unrestored variables'.format(len(unrestored_vars)))

        self.init_unrestored_op = (tf.variables_initializer(unrestored_vars)
                                   if unrestored_vars else tf.no_op())

        self.saver = tf.train.Saver(**kwargs_saver)

    @staticmethod
    def get_var_list_of_ckpt_dir(ckpt_dir, skip_var_names=None):
        all_saveable_objects = tf_helpers.all_saveable_objects()
        assert len(all_saveable_objects) > 0, 'No saveable objects in graph!'
        all_names_to_restore = VarNames(ckpt_dir).read(skip_var_names)
        return [v for v in all_saveable_objects if v.name in all_names_to_restore]

    def save(self, sess, global_step):
        self.saver.save(sess, self.ckpt_base_file_path, global_step)

    def restore_at_itr(self, sess, restore_itr=-1):
        """ Restores variables and initialized un-restored variables. """
        ckpt_to_restore_itr, ckpt_to_restore = self.get_latest_checkpoint_before_itr(restore_itr)
        assert ckpt_to_restore is not None
        self.restore_ckpt(sess, ckpt_to_restore)
        return ckpt_to_restore_itr

    def restore_ckpt(self, sess, ckpt_to_restore):
        self.saver.restore(sess, ckpt_to_restore)
        sess.run(self.init_unrestored_op)

    def get_latest_checkpoint_before_itr(self, itr):
        all_ckpts_with_iterations = Saver.all_ckpts_with_iterations(self.ckpt_dir)
        ckpt_to_restore_idx = -1 if itr == -1 \
            else Saver.index_of_ckpt_with_iter(all_ckpts_with_iterations, itr)
        ckpt_to_restore_itr, ckpt_to_restore = all_ckpts_with_iterations[ckpt_to_restore_idx]
        assert ckpt_to_restore is not None
        return ckpt_to_restore_itr, ckpt_to_restore

    @staticmethod
    def all_ckpts_with_iterations(ckpt_dir):
        return sorted(
            (Saver.iteration_of_checkpoint(ckpt_path), ckpt_path)
            for ckpt_path in Saver.all_ckpts_in(ckpt_dir))

    @staticmethod
    def index_of_ckpt_with_iter(ckpts_with_iterations, target_ckpt_itr):
        """ given a ascending list `ckpts_with_iterations` of (ckpt_iter, ckpt_path), returns the smallest index i in
        that list where target_ckpt_itr >= ckpt_iter """
        for i, (ckpt_iter, _) in reversed(list(enumerate(ckpts_with_iterations))):
            if target_ckpt_itr >= ckpt_iter:
                return i
        raise ValueError('*** Cannot find ckpt with iter <= {} in {}'.format(
            target_ckpt_itr, ckpts_with_iterations))

    @staticmethod
    def iteration_of_checkpoint(ckpt_path):
        ckpt_file_name = os.path.basename(ckpt_path)
        m = re.search(r'-(\d+)', ckpt_file_name)
        assert m is not None, 'Expected -(\\d+), got {}'.format(ckpt_path)
        return int(m.group(1))

    @staticmethod
    def all_ckpts_in(save_dir):
        return set(
            os.path.join(save_dir, os.path.splitext(fn)[0])
            for fn in os.listdir(save_dir)
            if _CKPT_FN in fn)

