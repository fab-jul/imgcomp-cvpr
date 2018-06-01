from os import path
import numpy as np
import scipy.misc
from PIL import Image


class ImagesIterator(object):
    """
    Iterates over images but keeps a cache. This speeds up iteration considerably because no disc access is needed.
    """
    class CachedImageLoader(object):
        def __init__(self, images_paths, pad, cache_max_mb):
            # Each pixel of an image takes around 3 uint8's when loaded, i.e., 3 bytes. Assuming all images in a
            # dataset are roughly of the same size (which is the case for most standard datasets), this gives an
            # estimate of the total number of bytes needed to cache all images of the dataset.
            self.images_paths = images_paths
            self.pad = pad
            self.cache_max_mb = cache_max_mb

            num_pixels_first_img = np.prod(Image.open(images_paths[0]).size)
            total_num_bytes = len(images_paths) * num_pixels_first_img * 3

            use_cache = total_num_bytes <= cache_max_mb * 1000 * 1000
            if use_cache:
                print('Using cache to keep {} images in memory...'.format(len(images_paths)))
            self.cache = ([None] * len(images_paths)) if use_cache else None

        def get(self, idx):
            if self.cache and self.cache[idx] is not None:
                return self.cache[idx]

            im = scipy.misc.fromimage(Image.open(self.images_paths[idx]))
            im, _ = self.add_padding(im)
            im = np.transpose(im, (2, 0, 1))  # make CHW
            if self.cache:
                self.cache[idx] = im
            return im

        def add_padding(self, im):
            # TODO: use undo pad when saving images to disk
            w, h, chan = im.shape
            if chan == 4:
                print('*** Ditching alpha channel...')
                return self.add_padding(im[:, :, :3])
            if w % self.pad == 0 and h % self.pad == 0:
                return im, lambda x: x

            wp = (self.pad - w % self.pad) % self.pad
            hp = (self.pad - h % self.pad) % self.pad
            wp_left = wp // 2
            wp_right = wp - wp_left
            hp_left = hp // 2
            hp_right = hp - hp_left
            paddings = [[wp_left, wp_right], [hp_left, hp_right], [0, 0]]
            im = np.pad(im, paddings, mode='constant')

            def _undo_pad(img_data_):
                return img_data_[wp_left:(-wp_right or None), hp_left:(-hp_right or None), :]
            return im, _undo_pad

        def __iter__(self):
            return (self.get(i) for i in range(len(self.images_paths)))

        def __str__(self):
            return "CachedImageLoader: " + (
                'No cache' if self.cache is None else
                '{} Images in cache; Max: {}MB'.format(
                        sum(1 for el in self.cache if el is not None), self.cache_max_mb))

    def __init__(self, images_paths, dataset_name, cache_max_mb):
        assert len(images_paths) > 0, 'No images!'
        self.images_paths = images_paths
        self.dataset_name = dataset_name
        self.cache_max_mb = cache_max_mb
        self.cached_image_loader = None  # Set on first call to iter_imgs because it depends on pad

    def iter_imgs(self, pad):
        """
        :param pad:
        :yield: (img_name, padded img_content, shape = CHW)
        """
        if self.cached_image_loader is None or self.cached_image_loader.pad != pad:
            print('Creating CachedImageLoader...')
            self.cached_image_loader = ImagesIterator.CachedImageLoader(self.images_paths, pad, self.cache_max_mb)
        return zip(map(path.basename, self.images_paths), self.cached_image_loader)

    def __str__(self):
        return 'Dataset {}, {} paths'.format(self.dataset_name, len(self.images_paths))
