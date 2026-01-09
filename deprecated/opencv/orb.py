import numpy as np
import cv2
from matplotlib import pyplot as plt

orig = cv2.resize(cv2.imread('zebras-small.jpg',0), (400, 400))
#photo = cv2.resize(cv2.imread('photo-small.jpg',0), (400, 400))
photo = cv2.resize(cv2.imread('photo-cropped.jpg',0), (400, 400))

# Initiate ORB detector
orb = cv2.ORB.create()

kp1, des1 = orb.detectAndCompute(orig, None)
kp2, des2 = orb.detectAndCompute(photo, None)

bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

matches = bf.match(des1, des2)
matches = sorted(matches, key = lambda x:x.distance)
print(len(matches))

img3 = cv2.drawMatches(orig,kp1,photo,kp2,matches[:10],None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
plt.imshow(img3),plt.show()

#img1 = cv2.drawKeypoints(orig, kp1, None, color=(0,255,0), flags=0)
#img2 = cv2.drawKeypoints(photo, kp2, None, color=(0,255,0), flags=0)
#plt.imshow(img1), plt.show()
#plt.imshow(img2), plt.show()