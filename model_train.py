import argparse
from ultralytics import YOLO

def train_model(data_path, model_name):
    # Load the model
    model = YOLO("yolov8n.yaml")  # build a new model from YAML
    model = YOLO("yolov8n.pt")  # load a pretrained model (recommended for training)
    model = YOLO("yolov8n.yaml").load("yolov8n.pt")  # build from YAML and transfer weights

    # Making this so the results are saved to a semi-resonable name
    results_name = model_name + "_results" 

    # Train the model
    results = model.train(data=data_path, epochs=100, imgsz=640, name=results_name)

    # Print training results
    print(results)

    # Evaluate the model
    metrics = model.val()

    # Print evaluation metrics
    print("Model evaluation metrics:")
    print(metrics)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a YOLO model with a custom dataset.")
    parser.add_argument("data_path", type=str, help="Path to the data.yaml file")
    parser.add_argument("model_name", type=str, help="Name for the saved model")

    args = parser.parse_args()

    train_model(args.data_path, args.model_name)
