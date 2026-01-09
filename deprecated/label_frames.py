import cv2, argparse
from pathlib import Path


def process(img):
  img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # convert to grayscale
  _, thresh = cv2.threshold(img_gray, 163, 255, cv2.THRESH_BINARY) # threshhold (convert to just solid black and white)
  img_canny = cv2.Canny(thresh, 0, 0) # canny edge detection
  # dilate expands shapes, erode shrinks them. https://docs.opencv.org/3.4/db/df6/tutorial_erosion_dilatation.html
  img_dilate = cv2.dilate(img_canny, None, iterations=7)
  return cv2.erode(img_dilate, None, iterations=7)


def get_contours(img, min_area = 150):
  # flags determine whether to return nested contours (external = only top level contours), and whether to combine contours
  contours, _ = cv2.findContours(process(img), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

  for c in contours:
    if cv2.contourArea(c) > min_area:
      yield c


def crop_contour(img, contour, crop_width=640, crop_height=480):
  x, y, w, h = cv2.boundingRect(contour)
  img_width = img.shape[1]
  img_height = img.shape[0]

  # find centre of bounding rectangle
  cx = x + w//2
  cy = y + h//2

  # clamp center so that output will have at least width x height pixels
  crop_cx = max(cx, crop_width//2)
  crop_cy = max(cy, crop_height//2)

  crop_cx = min(crop_cx, img_width - crop_width//2)
  crop_cy = min(crop_cy, img_height - crop_height//2)

  # move center to its coordinates in the cropped image
  if cx > crop_width//2:
    if cx > img_width - crop_width//2:
      cx -= img_width - crop_width
    else:
      cx = crop_width//2

  if cy > crop_height//2:
    if cy > img_height - crop_height//2:
      cy -= img_height - crop_height
    else:
      cy = crop_height//2

  # crop image
  return img[(crop_cy - crop_height//2):(crop_cy + crop_height//2),(crop_cx - crop_width//2):(crop_cx + crop_width//2)], (cx, cy)


def draw_menu(img, selected = None):
  # draw selected option in green
  if selected == 1:
    cv2.putText(img, "1: zebra", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255,0,255), thickness=2)
  else:
    cv2.putText(img, "1: zebra", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255,255,0), thickness=2)

  if selected == 2:
    cv2.putText(img, "2: aruco", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255,0,255), thickness=2)
  else:
    cv2.putText(img, "2: aruco", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255,255,0), thickness=2)

  if selected == 3:
    cv2.putText(img, "3: unknown", (20, 90), cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255,0,255), thickness=2)
  else:
    cv2.putText(img, "3: unknown", (20, 90), cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255,255,0), thickness=2)

  if selected == 4:
    cv2.putText(img, "4: skip", (20, 120), cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255,0,255), thickness=2)
  else:
    cv2.putText(img, "4: skip", (20, 120), cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255,255,0), thickness=2)
  
  cv2.putText(img, "SPACE: next", (420, 30), cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255,255,0), thickness=2)
  cv2.putText(img, "Q: quit", (500, 60), cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255,255,0), thickness=2)


def menu_loop(img):
  selected = None

  while True:
    draw_menu(img, selected)
    cv2.imshow("name", img)
    key = cv2.waitKey(0)

    if key == 32: # space = go to next
      if selected != None:
        return selected
    elif key == ord("1"): # zebra
      selected = 1
    elif key == ord("2"): # aruco
      selected = 2
    elif key == ord("3"): # unknown
      selected = 3
    elif key == ord("4"): # skip
      selected = 4
    elif key == ord("q"):
      raise KeyboardInterrupt("Q pressed, stopping.")


def main():
  parser = argparse.ArgumentParser()
  # args: input output_folder frame_skip
  parser = argparse.ArgumentParser()
  parser.add_argument("video", help="input video file")
  parser.add_argument("dest", help="output folder")

  parser.add_argument("--skip", "-s", help="amount of frames to skip", type=int, default=15)
  parser.add_argument("--area", "-a", help="minimum area of contours", type=int, default=1000)
  parser.add_argument("--width", help="cropped image width", type=int, default=640)
  parser.add_argument("--height", help="cropped image height", type=int, default=480)

  args = parser.parse_args()

  capture = cv2.VideoCapture(args.video)
  DEST = args.dest + "/"
  FRAME_SKIP = args.skip
  MIN_AREA = args.area
  CROP_WIDTH = args.width
  CROP_HEIGHT = args.height

  frame_count = 0
  image_count = 0

  try:
    while capture.isOpened():
      # skip frames - only works on video?
      # quite slow for some reason
      capture.set(cv2.CAP_PROP_POS_FRAMES, frame_count)
      frame_count += FRAME_SKIP

      ret, frame = capture.read()
      if not ret:
        break
      for c in get_contours(frame, min_area=MIN_AREA):
        x, y, w, h = cv2.boundingRect(c)
  
        cropped, cent = crop_contour(frame, c, crop_width=CROP_WIDTH, crop_height=CROP_HEIGHT)
        print(cent)

        # draw recange on copy of image (don't include this in training data)
        rect_img = cropped.copy()
        cv2.rectangle(rect_img, (cent[0] - w//2, cent[1] - h//2), (cent[0] + w//2, cent[1] + h//2), (0, 0, 255), 2)

        option = menu_loop(rect_img)

        if option != 4: # 4 == skip
          prefix = Path(args.video).stem
          filename = f"{prefix}-image-{image_count}"
          # write image
          cv2.imwrite(DEST + filename + ".jpg", cropped)
          # write image label
          label = option - 1
          with open(DEST + filename + ".txt", "w") as f:
            f.write(f"{label} {cent[0] / CROP_WIDTH} {cent[1] / CROP_HEIGHT} {w / CROP_WIDTH} {h / CROP_HEIGHT}")
          image_count += 1
  finally:
    capture.release()

if __name__ == "__main__":
  main()