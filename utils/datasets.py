# import numpy as np
import tensorflow_addons as tfa
import tensorflow_io as tfio
import tensorflow as tf
import os, sys
from random import shuffle
# import warnings
import matplotlib.pyplot as plt
sys.path.append('..')
from config import *

# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
# tf.config.threading.set_inter_op_parallelism_threads(6)
# tf.config.threading.set_intra_op_parallelism_threads(6)

def _show(image, label=None):
    plt.figure()
    plt.imshow(image)
    if label is not  None:
        plt.title(str(label.numpy()))
    plt.axis('off')
    plt.show()


class MS1MRecognition:

    def __init__(self, tfrecord_path=None, image_size=IMAGE_SIZE, image_dir=None, argument=True,
                 batch_size=BATCH_SIZE, output_filename=False,
                 ):
        self._num_classes = 85742
        self._num_classes_reserved = 100
        self._tfrecords_path = tfrecord_path
        # self._read_annos()
        self._shuffle_length = 1024
        self._image_dir = image_dir
        self._image_size = image_size
        self._argument = argument
        self._output_filename = output_filename
        self._batch_size = batch_size

        self.raw_image_dataset = None
        self.parsed_image_dataset = None
        self.decoded_image_dataset = None
        self.batch_dataset = None
        self.shuffle_image_dataset = None

        self.read_buffer_size = 256 * (1024 * 1024)
        self.shuffle_buffer_size = 2048
        self.prefetch_buffer_size = 8

        self._image_feature_description = {
            'label': tf.io.FixedLenFeature([], tf.int64),
            'filepath': tf.io.FixedLenFeature([], tf.string),
            'image_raw': tf.io.FixedLenFeature([], tf.string),
        }

    @staticmethod
    def _encode_image(image):
        return tf.image.encode_jpeg(image)

    @staticmethod
    def _decode_image_raw(image_raw):
        return tf.image.decode_jpeg(image_raw)

    @staticmethod
    def _random_blur(image):
        _th = tf.cast(0.5, tf.float32)

        def _return_blur_image():
            b_image = tfio.experimental.filter.gaussian([image], 3, 0)
            return tf.reshape(b_image, (IMAGE_SIZE, IMAGE_SIZE, 3))

        def _return_orignal_image():
            return image

        return tf.cond(tf.greater(tf.random.uniform([], maxval=1., dtype=tf.float32), _th),
                       _return_orignal_image, _return_blur_image)

    @staticmethod
    def _random_gray(image):
        _th = tf.cast(0.5, tf.float32)

        def _return_gray_image():
            g_image = tf.image.rgb_to_grayscale(image)
            rgb_image = tf.image.grayscale_to_rgb(g_image)
            return rgb_image

        def _return_rgb_image():
            return image

        return tf.cond(tf.greater(tf.random.uniform([], maxval=1., dtype=tf.float32), _th),
                       _return_rgb_image, _return_gray_image)

    def _apply_image_transform(self, image, label=None):
        if ROTATE > 0:
            image = tfa.image.rotate(image, angles=tf.random.uniform(shape=[], minval=-ROTATE, maxval=ROTATE))
        if RESIZE > 0:
            image = tf.image.resize(image, (RESIZE, RESIZE))
        if CROP > 0:
            image = tf.image.random_crop(image, [CROP, CROP, 3])
        image = tf.image.resize(image, (IMAGE_SIZE, IMAGE_SIZE))
        if FLIP:
            image = tf.image.random_flip_left_right(image)# original: -20~20
        if BRIGHTNESS > 0.:
            image = tf.image.random_brightness(image, BRIGHTNESS)  # original: 0.1
        if HUE > 0.:
            image = tf.image.random_hue(image, HUE)  # origianl: 0.03
        if CONTRAST > 0.:
            image = tf.image.random_contrast(image, 1. - CONTRAST, 1. + CONTRAST)  # original: 0.9~1.1
        if SATURATION > 0.:
            image = tf.image.random_saturation(image, 1. - SATURATION, 1. + SATURATION)  # original: 0.9~1.1
        if GRAY:
            return self._random_gray(image)
        if BLUR > 1:
            assert BLUR <= 3
            return self._random_blur(image)

        if label is None:
            return image
        else:
            return image, label

    @staticmethod
    def _bytes_feature(value):
        """Returns a bytes_list from a string / byte."""
        if isinstance(value, type(tf.constant(0))):
            value = value.numpy()  # BytesList won't unpack a string from an EagerTensor.
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

    @staticmethod
    def _int64_feature(value):
        """Returns an int64_list from a bool / enum / int / uint."""
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

    def _image_example(self, image_bytes, label, filename_bytes):

        feature = {
            'label': self._int64_feature(label),
            'filepath': self._bytes_feature(filename_bytes),
            'image_raw': self._bytes_feature(image_bytes),
        }

        return tf.train.Example(features=tf.train.Features(feature=feature))

    def _parse_image_function(self, example_proto):
        return tf.io.parse_single_example(example_proto, self._image_feature_description)

    def _map_single_binary_image(self, image_features):
        label = image_features['label']
        filename = image_features['filepath']
        image_raw = image_features['image_raw']
        image = self._decode_image_raw(image_raw)
        image = tf.cast(image, tf.float32)
        image_id = tf.one_hot(label, NUM_CLASSES, dtype=tf.float32)

        if self._argument:
            image = self._apply_image_transform(image)
        image = tf.image.resize(image, size=(self._image_size, self._image_size))

        image /= 255.0
        # image -= 0.5
        # image *= 2

        if self._output_filename:
            return image, image_id, filename
        else:
            return image, image_id

    def tfrecords_to_dataset(self):
        self.raw_image_dataset = tf.data.TFRecordDataset(self._tfrecords_path,
                                                         buffer_size=self.read_buffer_size)
        self.shuffle_image_dataset = self.raw_image_dataset.shuffle(self.shuffle_buffer_size)
        self.parsed_image_dataset = self.shuffle_image_dataset.map(self._parse_image_function)
        self.decoded_image_dataset = self.parsed_image_dataset.map(self._map_single_binary_image,
                                                                   num_parallel_calls=6)
        self.batch_dataset = self.decoded_image_dataset.batch(self._batch_size,
                                                              drop_remainder=True).prefetch(self.prefetch_buffer_size)

        return self.batch_dataset

    def imagefile_to_tfrecords(self):
        num_classes = self._num_classes - self._num_classes_reserved
        files = []
        for d in range(num_classes):
            d_ = str(d)
            filedir = os.path.join(self._image_dir, d_)
            fs = os.listdir(filedir)
            for f in fs:
                files.append(d_ + '/' + f)
        shuffle(files)
        cnt = 0
        num = len(files)
        with tf.io.TFRecordWriter(self._tfrecords_path) as writer:
            for file in files:
                label = eval(file.split('/')[0])
                image_string = open(os.path.join(self._image_dir, file), 'rb').read()
                tf_example = self._image_example(image_string, label, file.encode())
                writer.write(tf_example.SerializeToString())
                cnt += 1
                if cnt % 10000 == 0:
                    print("%dM images processed. %.3f complete." % (cnt // 10000, cnt / num))




if __name__ == '__main__':

    data = MS1MRecognition(
        tfrecord_path=TFRECORDS_PATH,
        image_size=112,
        image_dir=IMAGE_ROOT_DIR,
        output_filename=True,
        argument=True,
        batch_size=8,
    )
    data.imagefile_to_tfrecords()
    # samples = data.tfrecords_to_dataset()
    # for sample in samples:
    #     image0, label0, filename0 = sample
    #     for img, lb, fn in zip(image0, label0, filename0):
    #         # break
    #         img = img.numpy()
    #         # print(img.min(), img.max())
    #         print(fn, lb)
    #         # img /= 2.0
    #         # img += 0.5
    #         _show(img)
