def nothing(x):
    i = 0

    
import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('C:/Users/ashwi/Desktop/opencv/opencv.png')
img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

cv2.namedWindow('image',cv2.WINDOW_NORMAL)
cv2.resizeWindow('image',1000,1000)
cv2.createTrackbar('Blur','image',5,200, nothing )

while True:
    n = cv2.getTrackbarPos('Blur','image')
    if(n > 1):
        kernel = np.ones((n,n),np.float32)/(n*n)
        dst = cv2.filter2D(img,-1,kernel)
        cv2.imshow('image',dst)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
