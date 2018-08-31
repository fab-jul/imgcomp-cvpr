import bit_counter as bc
import probclass


class BppFetcher(object):
    """
    Fetch real bpp for some symbols volume
    """
    def __init__(self, pred: probclass.PredictionNetwork, checker: probclass.ProbclassNetworkTesting):
        self.pred = pred
        self.checker = checker

    def get_bpp(self, symbols, num_pixels):
        """
        :param symbols: all symbols of an image, NCHW, ndarray
        :param num_pixels: int num pixels in image
        :return: tuple (bpp_real, bpp_theory), where
                    bpp_real: the bpp used when encoding to a file,
                    bpp_theory: the bpp as reported with entropy(softmax), which should match the entropy loss.
                See val.py
        """
        assert symbols.ndim == 4
        bpp = bc.encode_decode_to_file_ctx(symbols, self.pred, syms_format='CHW', verbose=True) / num_pixels
        bpp_theory = self.checker.get_total_bit_cost(symbols) / num_pixels
        return bpp, bpp_theory


def num_pixels_in_image(im):
    c, h, w = im.shape
    assert c == 3, 'Expected RGB image, got {}'.format(im.shape)
    return w * h
