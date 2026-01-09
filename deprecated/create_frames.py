import cv2
import os
import argparse

def extract_frames(video_path, output_folder, skip_frames):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"Created directory {output_folder}")
        
    # Extract video file name without extension to use in frame filenames
    video_filename = os.path.basename(video_path)
    video_filename_no_ext = os.path.splitext(video_filename)[0]

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    frame_count = 0  # Counts every frame
    saved_count = 0  # Counts saved frames
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Save the frame if the current count modulo skip_frames is 0
        if frame_count % skip_frames == 0:
            frame_filename = os.path.join(output_folder, f'{video_filename_no_ext}_frame_{saved_count:04d}.png')
            cv2.imwrite(frame_filename, frame)
            saved_count += 1

        frame_count += 1

    cap.release()
    print(f"Done extracting frames. Extracted {saved_count} frames out of {frame_count} total frames.")

def main():
    # Example usage -- python create_frames.py "drone-videos\DJI_0234.MP4" "frames" 25
    parser = argparse.ArgumentParser(description="Extract frames from video file")
    parser.add_argument("video_path", type=str, help="Path to the video file")
    parser.add_argument("output_folder", type=str, help="Path to the output folder where frames will be saved")
    parser.add_argument("skip_frames", type=int, help="Number of frames to skip")

    args = parser.parse_args()

    extract_frames(args.video_path, args.output_folder, args.skip_frames)

if __name__ == "__main__":
    main()
