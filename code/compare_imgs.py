import argparse
import numpy as np
import glob
import os
import scipy.misc
import scipy.ndimage
from skimage.measure import compare_psnr
from skimage.measure import compare_ssim
from ms_ssim_np import MultiScaleSSIM
import functools


# speed test
# using batched input: 2.36s
# per image: 1.54s
def main(flags):
    import time
    inp_img_ps = sorted(glob.glob(flags.inp_glob))[:30]
    out_img_ps = sorted(glob.glob(flags.out_glob))[:30]
    assert len(inp_img_ps) == len(out_img_ps)

    inp = np.stack([_read_if_not_array(ip) for ip in inp_img_ps], 0)
    out = np.stack([_read_if_not_array(ip) for ip in out_img_ps], 0)

    m = []
    t = []
    for n in range(inp.shape[0]):
        s = time.time()
        v = compare_msssim(make_batched(inp[n, ...]), make_batched(out[n, ...]))
        t.append(time.time() - s)
        m.append(v)
    print(np.mean(m))
    print(np.sum(t))

    s = time.time()
    print(compare_msssim(inp, out))
    print(time.time() - s)


make_batched = functools.partial(np.expand_dims, axis=0)


def calc_and_print_ssim_and_psnr(inp_img_ps, out_img_ps):
    for inp_img, out_img in zip(inp_img_ps, out_img_ps):
        print(compare(inp_img, out_img))


def _read_if_not_array(im):
    if not isinstance(im, np.ndarray):
        assert os.path.exists(im)
        return scipy.ndimage.imread(im)
    return im

def compare_msssim(inp_img_batched, out_img_batched):
    return MultiScaleSSIM(inp_img_batched, out_img_batched)


def compare(inp_img, out_img, calc_ssim=True, calc_msssim=True, calc_psnr=True):
    inp_img = _read_if_not_array(inp_img)
    out_img = _read_if_not_array(out_img)

    assert inp_img.shape == out_img.shape

    def get_ssim():
        return compare_ssim(inp_img, out_img, multichannel=True, gaussian_weights=True, sigma=1.5)

    def get_msssim():
        return MultiScaleSSIM(make_batched(inp_img), make_batched(out_img))

    def get_psnr():
        return compare_psnr(inp_img, out_img)

    def _run_if(cond, fn):
        return fn() if cond else None

    return _run_if(calc_ssim, get_ssim), _run_if(calc_msssim, get_msssim), _run_if(calc_psnr, get_psnr)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('inp_glob')
    p.add_argument('out_glob')
    main(p.parse_args())
