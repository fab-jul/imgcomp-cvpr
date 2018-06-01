import constants
from os import path
from glob import glob


KNOWN_DATASETS = {
    'kodak': path.join(constants.VALIDATION_DATASETS_ROOT, 'kodak', '*.png'),
    'testset': path.join(constants.VALIDATION_DATASETS_ROOT, 'imagenet_256_train_val_128x128__100', '*.png')
}


def get_image_paths(images):
    """
    :param images: may be
    - key in KNOWN_DATASETS
    - directory containing PNGs
    - glob matching image files
    :return: tuple (list of image paths, short name of dataset)
    """
    images_glob, dataset_name = _get_glob_and_name(images)
    images_paths = sorted(glob(images_glob))
    if len(images_paths) == 0:
        raise ValueError('Not matching any files: {}'.format(images_glob))
    return images_paths, dataset_name


def _get_glob_and_name(images):
    try:
        return KNOWN_DATASETS[images], images
    except KeyError:
        if '*' not in images:  # images might be a dir with .pngs
            images = path.join(images, '*.png')

        # images is probably a glob
        return images, get_path_component_before_glob(images)


def get_path_component_before_glob(p):
    """ Given some path ending in one or more components containing *, return the left-most non-empty component not
    containig *, e.g., /some/path/dir/*/*/*.png => dir """
    for comp in reversed(p.strip(path.sep).split(path.sep)):
        if '*' not in comp:
            return comp
    raise ValueError('No component without *: {}'.format(p))

