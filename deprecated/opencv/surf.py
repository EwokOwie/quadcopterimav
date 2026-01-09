import cv2

img = cv2.imread("zebras-small.jpg", cv2.IMREAD_GRAYSCALE)

# need to use opencv-contrib-python for this
surf = cv2.xfeatures2d.SURF.create(400)

kp, des = surf.detectAndCompute(img,None)

print(len(kp))

#img2 = cv2.drawKeypoints(img,kp,None,(255,0,0),4)

#plt.imshow(img2),plt.show()