import cv2
import os
import glob
from sklearn.utils import shuffle
import numpy as np
import pandas as pd


class DataSet(object):
  _mode = 'rbg' # tryb rbg/gray
  _index_in_epoch = 0

  def next_batch(self, batch_size, train_path, image_size, classes, validation_size):

    images = []
    labels = []
    img_names = []
    cls = []

    ##zabawa w wczytanie części danych do pamięci

    start = self._index_in_epoch
    self._index_in_epoch += batch_size
    end = self._index_in_epoch

    for fields in classes:
        pathTest = os.path.join(train_path, fields, '*g')
        filesTest = len(glob.glob(pathTest))

        if filesTest < end:
            start = 0
            self._index_in_epoch = batch_size
            end = self._index_in_epoch


    for fields in classes:
        index = classes.index(fields)
        path = os.path.join(train_path, fields, '*g')
        files = glob.glob(path)[start:end]

        for fl in files:
            image = cv2.imread(fl)
            if image is None:
                image = cv2.imread(glob.glob(path)[1])

            #if(self._mode == 'gray'):
                #image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            image = cv2.resize(image, (image_size, image_size), 0, 0, cv2.INTER_LINEAR)
            #file = open('test2.txt', 'w')
            #file.write(pd.Series(np.reshape(image, 196608)).to_json(orient='values'))
            image = image.astype(np.float32)
            image = np.multiply(image, 1.0 / 255.0)

            #if (self._mode == 'gray'):
                #image = np.reshape(image, (image_size, image_size, 1))

            images.append(image)
            label = np.zeros(len(classes))
            label[index] = 1.0
            labels.append(label)
            flbase = os.path.basename(fl)
            img_names.append(flbase)
            cls.append(fields)
    images = np.array(images)
    labels = np.array(labels)
    cls = np.array(cls)

    images, labels,  cls = shuffle(images, labels, cls)

    if isinstance(validation_size, float):
        validation_size = int(validation_size * images.shape[0])

    validation_images = images[:validation_size]
    validation_labels = labels[:validation_size]
    validation_cls = cls[:validation_size]

    train_images = images[validation_size:]
    train_labels = labels[validation_size:]
    train_cls = cls[validation_size:]

    return train_images, train_labels, train_cls, \
           validation_images, validation_labels, validation_cls
