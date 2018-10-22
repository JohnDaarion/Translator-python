from pymongo import MongoClient
import cv2
import os
import glob
import numpy as np
from bson.binary import Binary
import pickle

client = MongoClient('192.168.99.100', 32768)

db = client['translator']
collection = db['images']
posts = db.posts

ala = db.collection_names(include_system_collections=False)
classes = ['a', 'b', 'c', '0']
train_path = 'training_data/rbg'

for fields in classes:
    index = classes.index(fields)
    path = os.path.join(train_path, fields, '*g')
    files = glob.glob(path)

    for fl in files:
        image = cv2.imread(fl)

        if image is None:
            image = cv2.imread(glob.glob(path)[1])

        image = cv2.resize(image, (256, 256), 0, 0, cv2.INTER_LINEAR)
        image = image.astype(np.float32)
        image = np.multiply(image, 1.0 / 255.0)
        post_id = posts.insert_one({'image': Binary(pickle.dumps(image, protocol=2), subtype=128), 'cls': fields})
        print('ala')

print('ala')
