import numpy as np
import cv2

import tensorflow as tf

def _aspect_preserving_resize(image, smallest_side):

    smallest_side = tf.convert_to_tensor(smallest_side, dtype=tf.int32)

    shape = tf.shape(image)

    height, width = shape[0], shape[1]
    height, width = tf.to_float(height), tf.to_float(width)

    smallest_side = tf.to_float(smallest_side)
    
    scale = tf.cond(
        tf.greater(height, width),
        lambda: smallest_side / width,
        lambda: smallest_side / height
    )

    new_height = tf.to_int32(height * scale)
    new_width = tf.to_int32(width * scale)

    resized_image = tf.image.resize_images(
        image, [new_height, new_width],
        tf.image.ResizeMethod.BILINEAR
    )
    resized_image.set_shape([None, None, 3])

    return resized_image


def parse_single_example(raw_str):

    features = {
        'image_raw': tf.FixedLenFeature([], tf.string),
        'label': tf.FixedLenFeature([], tf.int64)
    }
    example = tf.parse_single_example(raw_str, features=features)

    image = tf.image.decode_image(example['image_raw'], 3)
    image.set_shape([None, None, 3])
    label = tf.cast(example['label'], tf.int32)

    return image, label


def _umeyama(src, dst, estimate_scale):
    """Estimate N-D similarity transformation with or without scaling.
    Parameters
    ----------
    src : (M, N) array
        Source coordinates.
    dst : (M, N) array
        Destination coordinates.
    estimate_scale : bool
        Whether to estimate scaling factor.
    Returns
    -------
    T : (N + 1, N + 1)
        The homogeneous similarity transformation matrix. The matrix contains
        NaN values only if the problem is not well-conditioned.
    References
    ----------
    .. [1] "Least-squares estimation of transformation parameters between two
            point patterns", Shinji Umeyama, PAMI 1991, DOI: 10.1109/34.88573
    """

    num = src.shape[0]
    dim = src.shape[1]

    # Compute mean of src and dst.
    src_mean = src.mean(axis=0)
    dst_mean = dst.mean(axis=0)

    # Subtract mean from src and dst.
    src_demean = src - src_mean
    dst_demean = dst - dst_mean

    # Eq. (38).
    A = np.dot(dst_demean.T, src_demean) / num

    # Eq. (39).
    d = np.ones((dim,), dtype=np.double)
    if np.linalg.det(A) < 0:
        d[dim - 1] = -1

    T = np.eye(dim + 1, dtype=np.double)

    U, S, V = np.linalg.svd(A)

    # Eq. (40) and (43).
    rank = np.linalg.matrix_rank(A)
    if rank == 0:
        return np.nan * T
    elif rank == dim - 1:
        if np.linalg.det(U) * np.linalg.det(V) > 0:
            T[:dim, :dim] = np.dot(U, V)
        else:
            s = d[dim - 1]
            d[dim - 1] = -1
            T[:dim, :dim] = np.dot(U, np.dot(np.diag(d), V))
            d[dim - 1] = s
    else:
        T[:dim, :dim] = np.dot(U, np.dot(np.diag(d), V))

    if estimate_scale:
        # Eq. (41) and (42).
        scale = 1.0 / src_demean.var(axis=0).sum() * np.dot(S, d)
    else:
        scale = 1.0

    T[:dim, dim] = dst_mean - scale * np.dot(T[:dim, :dim], src_mean.T)
    T[:dim, :dim] *= scale

    return T


def cv_augmentation(img):
    pts = [30.2946, 65.5318, 48.0252, 33.5493, 62.7299,
                    51.6963, 51.5014, 71.7366, 92.3655, 92.2041];
    pts = np.asarray(pts, dtype="float32").reshape(2, -1).T

    pts_n = pts + np.random.randn(10).reshape(*pts.shape)*3
    T = _umeyama(pts_n, pts, True)

    scale = np.random.randn()*0.2 + 1
    shift = np.random.rand(2)*10 - 5
    R = cv2.getRotationMatrix2D((96/2, 112/2), 10 * np.random.randn() , scale)
    R = np.concatenate([R, [[0, 0, 1]]], axis=0).astype(R.dtype)

    T = np.dot(R, T)
    T[:2, 2] += shift

    n_img = cv2.warpAffine(img, T[:2], (96, 112))

    if np.random.randint(0, 1) == 0:
        n_img = n_img[:, ::-1, :]
    return (n_img.astype('float32') - 127.5) / 128.


def preprocess_for_train(height, width):

    def _preprocess(raw_str):
        image, label = parse_single_example(raw_str)

        resize_side = tf.random_uniform(
            [], minval=97,
            maxval=120+1, dtype=tf.int32
        )
        image = _aspect_preserving_resize(image, resize_side)
        image = tf.random_crop(image, [height, width, 3])

        # image = tf.py_func(cv_augmentation, [image], tf.float32, False)
        image = tf.py_func(lambda x: x, [image], tf.float32, False)
        image.set_shape([112, 96, 3])

        # # from 112x112 -> 112x96
        # image = image[:, 8:8+96, :]

        # image.set_shape([112, 96, 3])
        image = tf.image.random_flip_left_right(image)
        image = tf.image.adjust_brightness(image, 32./255.)

        # image = tf.image.resize_images(image, [height, width])
        image = (tf.to_float(image) - 127.5) / 128.

        return image, label

    return _preprocess


def preprocess_for_eval(height, width):

    def _preprocess(raw_str):
        image, label = parse_single_example(raw_str)
        image = tf.image.resize_images(image, [height, width])
        # image = image[:, 8:8+96, :]
        image.set_shape([112, 96, 3])
        image = (tf.to_float(image) - 127.5) / 128.
        return image, label

    return _preprocess
