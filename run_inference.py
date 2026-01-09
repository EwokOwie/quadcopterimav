import argparse
import os
from ultralytics import YOLO

def run_inference(model_path, images_dir, output_dir):
    # Load the YOLO model
    model = YOLO(model_path)


    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Run inference on the directory
    results = model.predict(images_dir, stream=True)

    # Process and save results
    for i, result in enumerate(results):
        # Save the prediction results to the output directory
        output_path = os.path.join(output_dir, f"prediction_{i}.jpg")
        result.save(output_path)
        print(f"Saved results to {output_path}")

if __name__ == "__main__":
    # Example usage 
    # python .\run_inference.py C:\Users\oweng\Documents\code\object-detection\runs\detect\Zebruco_v6_zebra_only_results\weights\best.pt C:\Users\oweng\Documents\code\object-detection\images\Zebruco.v7-test-images.yolov8\test\images predictions
    parser = argparse.ArgumentParser(description="Run YOLO inference on a directory of images and save predictions.")
    parser.add_argument("model_path", type=str, help="Path to the YOLO model file.")
    parser.add_argument("images_dir", type=str, help="Path to the directory containing test images.")
    parser.add_argument("output_dir", type=str, help="Path to the directory to save prediction images.")

    args = parser.parse_args()
    
    run_inference(args.model_path, args.images_dir, args.output_dir)
