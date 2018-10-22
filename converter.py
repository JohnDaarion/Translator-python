import cv2
import os


folders = ['testing_data', 'training_data']
classes = ['a', 'b', 'c', '0']

for folder in folders:
        for cls in classes:
                files = os.listdir(folder + '/rbg/' + cls)
                for file in files:
                        image = cv2.imread(folder + '/' + 'rbg/' + cls + '/' + file)
                        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                        cv2.imwrite(folder + '/' + 'gray/' + cls + '/' + 'GRAY ' + file, gray_image)
