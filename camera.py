import numpy as np
import cv2
import tensorflow as tf
import operator
from datetime import datetime

cap = cv2.VideoCapture(0)
session = tf.Session()
saver = tf.train.import_meta_graph('network-model.meta')
saver.restore(session, tf.train.latest_checkpoint('./'))
font = cv2.FONT_HERSHEY_SIMPLEX
bottomLeftCornerOfText = (10, 30)
fontScale = 0.5
fontColor = (255, 255, 0)
lineType = 2

mode = 'rbg' # tryb - obecnie rbg/gray

while (True):
    ret, image = cap.read()
    #frame = cv2.imread('15.jpg')
    #CBCR = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
    #lower = np.array([90, 130, 70])
    #upper = np.array([255, 180, 150])
    #mask = cv2.inRange(CBCR, lower, upper)
    # Bitwise-AND mask and original image
    #res = cv2.bitwise_and(frame, frame, mask=mask)
    i = str(datetime.utcnow()).replace(".", "").replace(" ", "").replace(":", " ").replace("-", " ")

    #if(mode == 'gray'):
        #image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    image = cv2.resize(image, (256, 256), 0, 0, cv2.INTER_LINEAR)
    cv2.imwrite(i + ".jpg", image)
    cv2.resizeWindow('frame', 512, 512)
    image = image.astype(np.float32)
    image = np.multiply(image, 1.0 / 255.0)
    j = 0
    images = np.reshape(image, (1, 256, 256, 3))
    z_pred = session.run("y_pred:0", feed_dict={"x:0": images})

    prediction = {'A': z_pred[0][0], 'B': z_pred[0][1], 'C': z_pred[0][2], '0': z_pred[0][3]}
    text = max(prediction.items(), key=operator.itemgetter(1))[0]
    image = cv2.resize(image, (512, 512), 0, 0, cv2.INTER_LINEAR)

    for val in prediction:

        pair = str(val) + ' : ' + str(prediction[val])
        y = 60 + 30 * j
        j += 1
        cv2.putText(image, pair, (10, y), font, fontScale,fontColor, lineType)

    cv2.putText(image, text, bottomLeftCornerOfText, font, fontScale * 2, fontColor, lineType)
    cv2.imshow('frame', image)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
