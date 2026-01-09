import cv2
import argparse
from ultralytics import YOLO

def process_video(video_path, model_path):
    # Load the specified YOLO model
    model = YOLO(model_path)

    # Run inference on the video
    results = model(video_path, stream=True)  # generator of Results objects

    # Define the video's resolution and frame rate
    width, height = 1920, 1080  # Video resolution
    fps = 59.94  # Frame rate

    # Setup video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Define the codec
    out = cv2.VideoWriter('annotated_video.mp4', fourcc, fps, (width, height))

    # Process each frame's results
    for result in results:
        frame = result.plot()  # Use plot() to get the frame with bounding boxes and labels
        out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))  # Convert frame from RGB to BGR format for OpenCV

    # Release the video writer
    out.release()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run YOLO model inference on a video and save the annotated video.")
    parser.add_argument("video_path", type=str, help="Relative Path to the video file")
    parser.add_argument("model_path", type=str, help="Relative Path to the YOLO model file")

    args = parser.parse_args()

    process_video(args.video_path, args.model_path)
