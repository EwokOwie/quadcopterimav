import cv2
# https://www.quora.com/How-can-I-detect-a-rectangle-using-OpenCV-code-in-Python

def detect_rect(frame):
  # Convert to grayscale 
  gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) 
  
  # Blur the image 
  blurred = cv2.GaussianBlur(gray, (5, 5), 0) 
  
  # Detect edges 
  edges = cv2.Canny(blurred, 50, 150) 
  
  # Find contours 
  contours, hierarchy = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) 
  
  # Filter contours 
  rects = [] 
  for contour in contours: 
      # Approximate the contour to a polygon 
      polygon = cv2.approxPolyDP(contour, 0.01 * cv2.arcLength(contour, True), True) 
      
      # Check if the polygon has 4 sides and the aspect ratio is close to 1 
      if len(polygon) == 4 and abs(1 - cv2.contourArea(polygon) / (cv2.boundingRect(polygon)[2] * cv2.boundingRect(polygon)[3])) < 0.1: 
          rects.append(polygon)

  # Draw rectangles 
  for rect in rects: 
    cv2.drawContours(frame, [rect], 0, (0, 255, 0), 2)
  
  return frame

# https://stackoverflow.com/questions/67457125/how-to-detect-white-region-in-an-image-with-opencv-python
def process(img):
  img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  _, thresh = cv2.threshold(img_gray, 163, 255, cv2.THRESH_BINARY)
  img_canny = cv2.Canny(thresh, 0, 0)
  img_dilate = cv2.dilate(img_canny, None, iterations=7)
  return cv2.erode(img_dilate, None, iterations=7)


def get_contours(img, min_area = 150):
  # flags determine whether to return nested contours (external = only top level contours), and whether to combine contours
  contours, _ = cv2.findContours(process(img), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

  for c in contours:
    if cv2.contourArea(c) > min_area:
      print(cv2.contourArea(c))
      yield c


def crop_biggest_contour(img, min_area = 150):
  SIZE = img.shape
  WIDTH = 640
  HEIGHT = 480

  contours, _ = cv2.findContours(process(img), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

  if contours:
    biggest = max(contours, key=cv2.contourArea)
    
    if cv2.contourArea(biggest) > min_area:
      x, y, w, h = cv2.boundingRect(biggest)

      # find centre
      cx = x + w//2
      cy = y + h//2

      cx = max(cx, WIDTH//2)
      cy = max(cy, HEIGHT//2)

      cx = min(cx, SIZE[1] - WIDTH//2)
      cy = min(cy, SIZE[0] - HEIGHT//2)

      # crop image
      res = img[(cy-HEIGHT//2):(cy + HEIGHT//2), (cx-WIDTH//2):(cx+WIDTH//2)]
      return res

  return img[0:HEIGHT, 0:WIDTH]


def crop_contour(img, contour, width=640, height=480):
  x, y, w, h = cv2.boundingRect(contour)

  # find centre of bounding rectangle
  old_cx = x + w//2
  old_cy = y + h//2
  print("old:", old_cx, old_cy)

  # clamp center so that output will have at least width x height pixels
  cx = max(old_cx, width//2)
  cy = max(old_cy, height//2)

  cx = min(cx, img.shape[1] - width//2)
  cy = min(cy, img.shape[0] - height//2)

  if old_cx > width//2:
    old_cx -= cx - width//2

  if old_cy > height//2:
    old_cy -= cy - height//2

  # crop image
  return img[(cy-height//2):(cy + height//2), (cx-width//2):(cx+width//2)], (old_cx, old_cy)