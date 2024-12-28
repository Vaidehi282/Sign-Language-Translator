import cv2 as cv

import matplotlib.pyplot as plt

img = cv.imread('vaidehi.png')
cv.imshow('vaidehi', img)
cv.waitKey(0)

plt.imshow(img)
plt.show()
