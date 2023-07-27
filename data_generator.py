import numpy as np
import keras
from tensorflow.keras.applications.resnet50 import preprocess_input
import cv2


class DataGenerator(keras.utils.Sequence):
    def __init__(self, paths_to_images, labels, batch_size=32, dim=(192, 192), n_channels=3,
                 n_classes=1, shuffle=True):
        self.dim = dim
        self.batch_size = batch_size
        self.labels = labels
        self.paths_to_images = paths_to_images
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.indexes = None
        self.on_epoch_end()

    def __len__(self):
        return int(np.floor(len(self.paths_to_images) / self.batch_size))

    def __getitem__(self, index):
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        temp_paths = [self.paths_to_images[k] for k in indexes]
        x, y = self.__data_generation(temp_paths)
        return x, y

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.paths_to_images))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __data_generation(self, temp_paths):
        x = np.empty((self.batch_size, *self.dim, self.n_channels))
        y = np.empty(self.batch_size, dtype=int)

        for i, path in enumerate(temp_paths):
            image = cv2.imread(path)
            x[i,] = cv2.resize(image, self.dim)
            y[i] = self.labels[path]

        x = preprocess_input(x)
        return x, y
