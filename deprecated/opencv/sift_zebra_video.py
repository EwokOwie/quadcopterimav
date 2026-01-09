# video analysis using opencv

import cv2
import time
import statistics

camera = cv2.VideoCapture(0)

sift = cv2.SIFT.create()
bf = cv2.BFMatcher()

#ZEBRA_PIC = cv2.imread('zebras-small.jpg',cv2.IMREAD_GRAYSCALE)
ZEBRA_PIC = cv2.imread('photo-cropped.jpg',cv2.IMREAD_GRAYSCALE)
ZEBRA_PIC = cv2.resize(ZEBRA_PIC, (800, 800))
#print("orig shape:", orig.shape)
ZEBRA_KP, ZEBRA_DES = sift.detectAndCompute(ZEBRA_PIC, None)

def detect(frame):
  frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
  frame = cv2.resize(frame, (800, 800))

  kp, des = sift.detectAndCompute(frame, None)

  matches = bf.knnMatch(ZEBRA_DES, des, k=2)

  good = []
  for m,n in matches:
    if m.distance < 0.75*n.distance:
      good.append([m])

  # show matches on image
  frame = cv2.drawMatchesKnn(ZEBRA_PIC, ZEBRA_KP, frame, kp, good,None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

  # show number of matches
  frame = cv2.putText(frame, f"M: {len(good)}", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255,255,0), thickness=2)

  return frame


def main():
  capture = cv2.VideoCapture('court.MP4')
  h, w = ZEBRA_PIC.shape
  #size = (int(capture.get(3))+w, max(int(capture.get(4)), h))
  #size = (int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
  size = (1600, 800)
  fps = capture.get(cv2.CAP_PROP_FPS)
  print(fps)

  # Define the codec and create VideoWriter object
  fourcc = cv2.VideoWriter_fourcc(*'DIVX')
  out = cv2.VideoWriter('output.avi', fourcc, fps, size)
  while capture.isOpened():
    ret, frame = capture.read()
    #print(frame, ret)
    if ret:
      frame = detect(frame)
      out.write(frame)
    else:
      break

  capture.release()
  out.release()

if __name__ == "__main__":
  main()