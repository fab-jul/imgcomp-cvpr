from saver import Saver


class RestoreManager(object):
    def __init__(self, ckpt_dir,
                 itr: int, continue_in_ckpt_dir: bool, from_identity: bool,
                 skip_var_names: str):
        if continue_in_ckpt_dir:
            print('Using restore dir as log dir!')
        self.ckpt_dir = ckpt_dir
        self.itr = itr
        self.continue_in_ckpt_dir = continue_in_ckpt_dir
        self.from_identity = from_identity
        self.skip_var_names = skip_var_names
        self.log_dir = Saver.log_dir_from_ckpt_dir(ckpt_dir)

    def restore(self, sess):
        skip_var_names = self.skip_var_names.split(',') if self.skip_var_names else []
        var_list = Saver.get_var_list_of_ckpt_dir(self.ckpt_dir, skip_var_names=skip_var_names)
        print('Restoring {} variables...'.format(len(var_list)))
        Saver(self.ckpt_dir, var_list=var_list).restore_at_itr(sess, restore_itr=self.itr)

    @staticmethod
    def from_flags(flags):
        """
        Parses --restore, --restore_itr, --restore_continue
        """
        if flags.from_identity:
            flags.restore = flags.from_identity
            flags.restore_skip_vars = 'global_step,Adam'
        if flags.restore is None:
            return None
        return RestoreManager(RestoreManager._get_restore_ckpt_dir(flags.restore),
                              flags.restore_itr,
                              flags.restore_continue,
                              flags.from_identity,
                              flags.restore_skip_vars)

    @staticmethod
    def _get_restore_ckpt_dir(restore_flag):
        if Saver.is_ckpt_dir(restore_flag):
            return restore_flag
        if Saver.is_ckpt_dir(Saver.ckpt_dir_for_log_dir(restore_flag)):
            return Saver.ckpt_dir_for_log_dir(restore_flag)
        raise ValueError('Invalid ckpt dir: {}'.format(restore_flag))
