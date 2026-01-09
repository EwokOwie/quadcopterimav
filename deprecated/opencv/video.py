import cv2
from orb_detect import detect
from rectangle_detect import *

#capture = cv2.VideoCapture("../../grass.MP4")
capture = cv2.VideoCapture("../../court.MP4")
fps = capture.get(cv2.CAP_PROP_FPS)
#out_size = (int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
#out_size = (800, 400) # two 400x400 images side-by-side
out_size = (1280, 480) # two 640x480 images
#out_size = (640, 480)

def main():
  # Define the codec and create VideoWriter object
  fourcc = cv2.VideoWriter_fourcc(*'DIVX')
  out = cv2.VideoWriter('output.avi', fourcc, fps, out_size)

  try:
    while capture.isOpened():
      ret, frame = capture.read()
      #print(frame, ret)
      if ret:
        new_frame = []
        for c in get_contours(frame):
          frame, _ = crop_contour(frame, c)
          # add to image
        out.write(frame)
        #get_contours(frame)
        #frame = crop_biggest_contour(frame)
        #frame = detect(frame)
        #cv2.imshow("output", frame)
        #cv2.waitKey(0)
        out.write(frame)
      else:
        break
  finally:
    capture.release()
    out.release()

if __name__ == "__main__":
  main()