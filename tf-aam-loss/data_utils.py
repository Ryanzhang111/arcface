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


def preprocess_for_train(height, width):

    def _preprocess(raw_str):
        image, label = parse_single_example(raw_str)

        resize_side = tf.random_uniform(
            [], minval=97,
            maxval=120+1, dtype=tf.int32
        )
        image = _aspect_preserving_resize(image, resize_side)
        image = tf.random_crop(image, [height, width, 3])
        image = tf.image.adjust_brightness(image, 32./255.)
        image = tf.image.random_flip_left_right(image)

        # image = tf.image.resize_images(image, [height, width])
        image = (tf.to_float(image) - 127.5) / 128.

        return image, label

    return _preprocess


def preprocess_for_eval(height, width):

    def _preprocess(raw_str):
        image, label = parse_single_example(raw_str)
        image = tf.image.resize_images(image, [height, width])
        image = (tf.to_float(image) - 127.5) / 128.
        return image, label

    return _preprocess
