import cv2
import copy
import pathlib
import warnings
import numpy as np
import tensorflow as tf


def unison_shuffled_copies(a, b):
    assert a.shape[-1] == b.shape[-1]
    p = np.random.permutation(a.shape[-1])
    return a[p], b[p]


class TFImageGenerator(tf.keras.utils.Sequence):

    def __init__(
            self,
            x_data_source,
            y_data_source,
            batch_size=32,
            x_shape=None,
            y_shape=None,
            # Normalization methods ['-1to1', '0to1']
            normalize=None,
            standardize: bool = False,
            shuffle: bool = True,
            shuffle_at_start: bool = True
    ):
        self.x_data_source, self.y_data_source = x_data_source, y_data_source
        self.x_data, self.y_data = self._fetch_all_data()
        self.batch_size = batch_size
        self.x_shape = x_shape
        self.y_shape = y_shape
        self.normalize = normalize
        self.standardize = standardize
        self.shuffle = shuffle
        if shuffle_at_start:
            self.on_epoch_end()

    def __len__(self):
        return int(self.x_data.shape[-1] // self.batch_size)

    def __getitem__(self, idx):
        batch_x = self.x_data[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = self.y_data[idx * self.batch_size:(idx + 1) * self.batch_size]

        batch_x_data = []
        batch_y_data = []
        for fidx in range(len(batch_x)):
            x_img = cv2.imread(str(batch_x[fidx]))
            y_img = cv2.imread(str(batch_y[fidx]))

            if self.x_shape != x_img.shape:
                hp = np.random.randint(0, x_img.shape[0] - self.x_shape[0])
                wp = np.random.randint(0, x_img.shape[1] - self.x_shape[1])

                x_img = x_img[hp:hp+self.x_shape[0], wp:wp+self.x_shape[1], :]
                y_img = y_img[hp:hp + self.x_shape[0], wp:wp + self.x_shape[1], :]

            batch_x_data.append(x_img)
            batch_y_data.append(y_img)

        if self.normalize is not None:
            batch_x_data = self.normalize_img(batch_x_data)
            # batch_y_data = self.normalize_img(batch_y_data)
        if self.standardize:
            batch_x_data = self.standardize_img(batch_x_data)
            # batch_y_data = self.standardize_img(batch_y_data)

        batch_x_data = np.array(batch_x_data, dtype=np.float32)
        batch_y_data = np.array(batch_y_data, dtype=np.float32)

        return batch_x_data, batch_y_data

    def __call__(self):
        for i in range(self.__len__()):
            yield self.__getitem__(i)

            if i == self.__len__() - 1:
                self.on_epoch_end()

    def _fetch_all_data(self):
        assert ((type(self.x_data_source) is pathlib.WindowsPath) & (type(self.y_data_source) is pathlib.WindowsPath)) | \
               ((type(self.x_data_source) is pathlib.Path) & (type(self.y_data_source) is pathlib.Path)) | \
               ((type(self.x_data_source) is np.ndarray) & (type(self.y_data_source) is np.ndarray)), \
            "Provide a numpy array or data path"

        if (type(self.x_data_source) is np.ndarray) & (type(self.y_data_source) is np.ndarray):
            warnings.warn('Make sure the data is structured and organized accordingly to user needs before'
                          'providing it to generator')
            x = copy.deepcopy(self.x_data_source)
            y = copy.deepcopy(self.y_data_source)
        else:
            # all_blurs = ['x_set_17_std-1p6', 'x_set_17_std-1p8', 'x_set_17_std-2p0',
            #              'x_set_17_std-2p2', 'x_set_17_std-2p4']
            x = np.array([])
            y = np.array([])
            # for blur_dir in all_blurs:
            #     x_subd = np.array([])
            allowed_ext = ('png', 'jpg', 'jpeg', 'bmp', 'ppm', 'tiff')

            for ext in allowed_ext:
                extension_related_x = list(pathlib.Path(self.x_data_source).glob(f'*.{ext}'))
                extension_related_y = list(pathlib.Path(self.y_data_source).glob(f'*.{ext}'))
                x = np.concatenate([x, extension_related_x])
                y = np.concatenate([y, extension_related_y])
                # x_subd = np.concatenate([x_subd, extension_related_x])
                # if blur_dir == all_blurs[-1]:
                #     y = np.concatenate([y, extension_related_y])

                # x_subd.sort()
            x.sort()
            y.sort()
                # x.append(x_subd)
        # x = np.array(x)
        return x, y

    def normalize_img(self, x):
        x = tf.cast(x, tf.float32)
        if self.normalize == '0to1':
            return x / 255.
        elif self.normalize == '-1to1':
            return x / 127.5 - 1.0
        else:
            raise NotImplementedError

    def standardize_img(self, x):
        return tf.image.per_image_standardization(x)

    # shuffles the dataset at the end of each epoch
    def on_epoch_end(self):
        self.x_data, self.y_data = unison_shuffled_copies(self.x_data, self.y_data)
