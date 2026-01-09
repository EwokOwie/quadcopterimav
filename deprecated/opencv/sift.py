import cv2

img = cv2.imread("zebras-small.jpg", cv2.IMREAD_GRAYSCALE)

sift = cv2.SIFT.create(100)

kp, des = sift.detectAndCompute(img, None)

print(len(kp))

img2 = cv2.drawKeypoints(img,kp,None,(255,0,0),4)

plt.imshow(img2),plt.show()