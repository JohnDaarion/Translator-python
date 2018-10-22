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

classes = ['a', 'b', 'c', '0']
train_path = 'training_data/rbg'

kot = pickle.loads(posts.find_one({'cls':'a'})['image'])

ala = db.collection_names(include_system_collections=False)
#npArray = pickle.loads(record['feature2'])
print('ala')
