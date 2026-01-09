# video analysis using opencv

import cv2
import time
import statistics

camera = cv2.VideoCapture(0)

orb = cv2.ORB.create()
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

ZEBRA_PIC = cv2.imread('zebras-small.jpg',0)
ZEBRA_PIC = cv2.resize(ZEBRA_PIC, (400, 400))
#print("orig shape:", orig.shape)
ZEBRA_KP, ZEBRA_DES = orb.detectAndCompute(ZEBRA_PIC, None)

def detect(frame):
  kp, des = orb.detectAndCompute(frame, None)

  matches = bf.match(ZEBRA_DES, des)
  #matches = sorted(matches, key = lambda x:x.distance)
  good = []
  for i in matches:
    if i.distance < 50:
      good.append(i)

  # show matches on image
  frame = cv2.drawMatches(ZEBRA_PIC, ZEBRA_KP, frame, kp, good,None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

  # show number of matches
  frame = cv2.putText(frame, f"M: {len(good)}", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255,255,0), thickness=2)

  return frame


def main():
  capture = cv2.VideoCapture('test.mp4')
  h, w = ZEBRA_PIC.shape
  size = (int(capture.get(3))+w, max(int(capture.get(4)), h))
  #size = (int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
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
