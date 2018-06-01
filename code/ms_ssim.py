import tensorflow as tf
import numpy as np


def gauss_kernel(sigma, size):
    # Adaptive kernel size based on sigma,
    # for fixed kernel size, hardcode N
    # truncate limits kernel size as in scipy's gaussian_filter
    N = size//2
    x = np.arange(-N, N + 1, 1.0)
    g = np.exp(-x * x / (2 * sigma * sigma))
    g = g / np.sum(np.abs(g))
    return g


def gaussian_blur(image, sigma, size, cdim=3,mode='VALID'):
    if sigma == 0:
        return image
    kernel = gauss_kernel(sigma, size)

    outputs = []

    kernel_size = kernel.shape[0]
    total_pad = max(kernel_size - image.shape.as_list()[2], 0)

    pad_w1 = total_pad + 1 // 2
    pad_w2 = total_pad // 2

    image = tf.pad(image, [[0, 0], [pad_w1, pad_w2], [pad_w1, pad_w2], [0, 0]], mode='REFLECT')

    for channel_idx in range(cdim):
        data_c = image[:, :, :, channel_idx:(channel_idx + 1)]
        g = np.expand_dims(kernel, 0)
        g = np.expand_dims(g, axis=2)
        g = np.expand_dims(g, axis=3)

        data_c = tf.nn.conv2d(data_c, g, [1, 1, 1, 1], mode)
        g = np.expand_dims(kernel, 1)
        g = np.expand_dims(g, axis=2)
        g = np.expand_dims(g, axis=3)
        data_c = tf.nn.conv2d(data_c, g, [1, 1, 1, 1], mode)
        outputs.append(data_c)
    return tf.concat(outputs, axis=3)


def kernel_blur(image, kernel, cdim=3,pad=True,mode='VALID'):
    outputs = []
    if pad:
        pad_w1 = (kernel.shape[0]-1) // 2
        pad_w2 = (kernel.shape[0]) // 2
        image = tf.pad(image, [[0, 0], [pad_w1, pad_w2], [pad_w1, pad_w2], [0, 0]], mode='REFLECT')
    for channel_idx in range(cdim):
        data_c = image[:, :, :, channel_idx:(channel_idx + 1)]
        g = np.expand_dims(kernel, 0)
        g = np.expand_dims(g, axis=2)
        g = np.expand_dims(g, axis=3)

        data_c = tf.nn.conv2d(data_c, g, [1, 1, 1, 1], mode)
        g = np.expand_dims(kernel, 1)
        g = np.expand_dims(g, axis=2)
        g = np.expand_dims(g, axis=3)
        data_c = tf.nn.conv2d(data_c, g, [1, 1, 1, 1], mode)
        outputs.append(data_c)
    return tf.concat(outputs, axis=3)


def _FSpecialGauss(size, sigma):
    """Function to mimic the 'fspecial' gaussian MATLAB function."""
    radius = size // 2
    offset = 0.0
    start, stop = -radius, radius + 1
    if size % 2 == 0:
        offset = 0.5
        stop -= 1
    x, y = np.mgrid[offset + start:stop, offset + start:stop]
    assert len(x) == size
    g = np.exp(-((x ** 2 + y ** 2) / (2.0 * sigma ** 2)))
    return g / g.sum()


def _SSIMForMultiScale(img1, img2, max_val=255, filter_size=11,
                       filter_sigma=1.5, k1=0.01, k2=0.03):
    _, height, width, _ = img1.shape.as_list()
    size = min(filter_size, height, width)
    sigma = size * filter_sigma / filter_size if filter_size else 0
    if filter_size:
        mu1 =     gaussian_blur(img1        , sigma, size)
        mu2 =     gaussian_blur(img2        , sigma, size)
        sigma11 = gaussian_blur(img1 * img1 , sigma, size)
        sigma22 = gaussian_blur(img2 * img2 , sigma, size)
        sigma12 = gaussian_blur(img1 * img2 , sigma, size)
    else:
        # Empty blur kernel so no need to convolve.
        mu1, mu2 = img1, img2
        sigma11 = img1 * img1
        sigma22 = img2 * img2
        sigma12 = img1 * img2
    mu11 = mu1 * mu1
    mu22 = mu2 * mu2
    mu12 = mu1 * mu2
    sigma11 -= mu11
    sigma22 -= mu22
    sigma12 -= mu12
    # Calculate intermediate values used by both ssim and cs_map.
    c1 = (k1 * max_val) ** 2
    c2 = (k2 * max_val) ** 2
    v1 = 2.0 * sigma12 + c2
    v2 = sigma11 + sigma22 + c2
    ssim = (((2.0 * mu12 + c1) * v1) / ((mu11 + mu22 + c1) * v2))
    ssim = tf.reduce_mean(ssim)
    cs = tf.reduce_mean(v1 / v2)
    return ssim,cs


def MultiScaleSSIM(img1, img2, max_val=255, filter_size=11, filter_sigma=1.5,
                   k1=0.01, k2=0.03, weights=None, data_format='NHWC', name=None):
    """Return the MS-SSIM score between `img1` and `img2`.

    This function implements Multi-Scale Structural Similarity (MS-SSIM) Image
    Quality Assessment according to Zhou Wang's paper, "Multi-scale structural
    similarity for image quality assessment" (2003).
    Link: https://ece.uwaterloo.ca/~z70wang/publications/msssim.pdf

    Author's MATLAB implementation:
    http://www.cns.nyu.edu/~lcv/ssim/msssim.zip

    Arguments:
        img1: Numpy array holding the first RGB image batch.
        img2: Numpy array holding the second RGB image batch.
        max_val: the dynamic range of the images (i.e., the difference between the
          maximum the and minimum allowed values).
        filter_size: Size of blur kernel to use (will be reduced for small images).
        filter_sigma: Standard deviation for Gaussian blur kernel (will be reduced
          for small images).
        k1: Constant used to maintain stability in the SSIM calculation (0.01 in
          the original paper).
        k2: Constant used to maintain stability in the SSIM calculation (0.03 in
          the original paper).
        weights: List of weights for each level; if none, use five levels and the
          weights from the original paper.

    Returns:
        MS-SSIM score between `img1` and `img2`.

    Raises:
        RuntimeError: If input images don't have the same shape or don't have four
          dimensions: [batch_size, height, width, depth].
    """
    if not (img1.shape.is_fully_defined() and img2.shape.is_fully_defined()):
        raise RuntimeError('Shapes must be fully defined (%s, %s)',
                           img1.shape, img2.shape)
    if img1.shape != img2.shape:
        raise RuntimeError('Input images must have the same shape (%s vs. %s).',
                           img1.shape, img2.shape)
    if len(img1.shape) != 4:
        raise RuntimeError('Input images must have four dimensions, not %d',
                           len(img1.shape))

    with tf.name_scope(name, 'ms-ssim'):
        if data_format == 'NCHW':
            img1, img2 = tf.transpose(img1, (0, 2, 3, 1)), tf.transpose(img2, (0, 2, 3, 1))
            data_format = 'NHWC'

        # Note: default weights don't sum to 1.0 but do match the paper / matlab code.
        weights = np.array(weights if weights else
                           [0.0448, 0.2856, 0.3001, 0.2363, 0.1333])
        levels = weights.size
        weights = tf.convert_to_tensor(weights,dtype=tf.float32)
        downsample_filter = np.ones((2,)) / 2.0
        im1, im2 = (img1,img2)
        mssim = []
        mcs = []
        for l in range(levels):
            ssim, cs = _SSIMForMultiScale(
                im1, im2, max_val=max_val, filter_size=filter_size,
                filter_sigma=filter_sigma, k1=k1, k2=k2)
            mssim.append(ssim)
            mcs.append(cs)
            filtered = [kernel_blur(im, downsample_filter, pad=True, mode='VALID')
                        for im in (im1, im2)]
            im1, im2 = [x[:, ::2, ::2, :] for x in filtered]
        mcs = tf.stack(list(mcs), axis=0)
        mssim = tf.stack(list(mssim), axis=0)

        return (tf.reduce_prod(mcs[0:levels - 1] ** weights[0:levels - 1]) *
                (mssim[levels - 1] ** weights[levels - 1]))

