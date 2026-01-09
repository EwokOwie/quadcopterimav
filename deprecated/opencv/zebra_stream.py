#!/usr/bin/python

# https://pyimagesearch.com/2020/12/21/detecting-aruco-markers-with-opencv-and-python/
# https://docs.opencv.org/4.x/d5/dae/tutorial_aruco_detection.html
# https://towardsdatascience.com/video-streaming-in-web-browsers-with-opencv-flask-93a38846fe00

# Steps to get this working:
# install dependencies (sudo apt-get install):
#       python3-flask
#       python-opencv
#
# transfer this file to the pi using sftp (using FileZilla)
# plug in webcam
# run the file with python
# enter the ip address of the pi in web browser, and use 5000 as the port (e.g. 192.168.1.16:5000)

from flask import Flask, Response, render_template_string
import cv2
import time
import statistics

app = Flask(__name__)
camera = cv2.VideoCapture(0)

orb = cv2.ORB.create()
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

orig = cv2.imread('zebras-small.jpg',0)
orig = cv2.resize(orig, (400, 400))
#print("orig shape:", orig.shape)
kp1, des1 = orb.detectAndCompute(orig, None)

# fps measurement
prev_time = time.time()
frame_times = [0]*10 # set length of frame average here

def capture_video():
  global prev_time
  
  while True:
    success, frame = camera.read()  # read the camera frame
    if not success:
      break
    else:
      resized = cv2.resize(frame, (400, 400))
      #print("resized shape:", resized.shape)
      kp2, des2 = orb.detectAndCompute(resized, None)

      matches = bf.match(des1, des2)
      #matches = sorted(matches, key = lambda x:x.distance)
      good = []
      for i in matches:
        if i.distance < 50:
          good.append(i)

      # show matches on image
      frame = cv2.drawMatches(orig,kp1,resized,kp2,good,None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
      
			# show fps
      new_time = time.time()
      frame_times.append(new_time - prev_time)
      frame_times.pop(0)
      fps = 1 / statistics.fmean(frame_times)
      prev_time = new_time
      frame = cv2.putText(frame, f"FPS: {int(fps)}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255,255,0), thickness=2)

      # show matches
      frame = cv2.putText(frame, f"M: {len(good)}", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255,255,0), thickness=2)

      ret, buffer = cv2.imencode('.jpg', frame)
      frame = buffer.tobytes()
      yield (b'--frame\r\n'
      b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')  # concat frame one by one and show result

@app.route("/")
def index():
  return render_template_string("""<body>
<div class="container">
    <div class="row">
        <div class="col-lg-8  offset-lg-2">
            <h3 class="mt-5">Live Streaming</h3>
            <img src="{{ url_for('video') }}" width="100%">
        </div>
    </div>
</div>
</body>""")

@app.route("/video")
def video():
  return Response(capture_video(), mimetype="multipart/x-mixed-replace; boundary=frame")

if __name__ == "__main__":
  try:
    # don't use debug=True, it opens video stream twice
    app.run(host="0.0.0.0") # expose to the network
  finally:
    camera.release()
