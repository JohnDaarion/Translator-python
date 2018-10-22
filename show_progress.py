import cv2
import os

while (True):
    file = os.listdir('plots/')[os.listdir('plots/').__len__() - 1]

    img = cv2.imread('plots/' + file)
    if not img is None:
        img = cv2.resize(img, (512, 512), 0, 0, cv2.INTER_LINEAR)
        cv2.resizeWindow('frame', 512, 512)
        cv2.imshow('Progress', img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
