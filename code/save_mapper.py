import tensorflow as tf
import os
import re
import sys
import numpy as np
from tensorflow.python import pywrap_tensorflow
from fjcommon import tf_helpers


def print_all_in_ckpt(ckpt_path):
    reader = pywrap_tensorflow.NewCheckpointReader(ckpt_path)
    var_to_shape_map = reader.get_variable_to_shape_map()
    for key in var_to_shape_map:
        print('{} {}\n'.format(key, reader.get_tensor(key).shape))
        #f.write('{}\n'.format(reader.get_tensor(key).shape))


def get_all_variable_names(ckpt_path):
    reader = pywrap_tensorflow.NewCheckpointReader(ckpt_path)
    var_to_shape_map = reader.get_variable_to_shape_map()
    return list(var_to_shape_map.keys())


def migrate_vidcompress_to_imgcomp(sess, ckpt_path, out_name):
    import pickle
    mapping = pickle.load(open('mapping.pkl', 'rb'))
    new_p = os.path.join(os.path.dirname(ckpt_path), out_name)

    create_new_ckpt_with_name_mapping(sess, ckpt_path, new_p, mapping)

    from fjcommon.functools_ext import snd
    new_var_names = [n + ':0' for n in map(snd, mapping)]
    var_names_p = os.path.join(os.path.dirname(ckpt_path), 'var_names.pkl')
    print('Dumping new var names {}...'.format(var_names_p))
    pickle.dump(new_var_names, open(var_names_p, 'wb'))

    print_all_in_ckpt(new_p)


def create_new_ckpt_with_name_mapping(sess, ckpt_path_in, ckpt_path_out, name_mapping):
    load_mapping = {}
    save_mapping = {}
    variables = tf_helpers.all_saveable_objects()
    name_without_device = lambda v_: v_.name.split(':')[0]  # remove ':0' from variable names
    variables_by_name = {name_without_device(v): v for v in variables}

    for from_name, to_name in name_mapping:
        load_mapping[from_name] = variables_by_name[to_name]
        save_mapping[to_name] = variables_by_name[to_name]

    load = tf.train.Saver(var_list=load_mapping)
    save = tf.train.Saver(var_list=save_mapping)

    print('Loading...')
    load.restore(sess, ckpt_path_in)

    print('Saving...')
    save.save(sess, ckpt_path_out)


