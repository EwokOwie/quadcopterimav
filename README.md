# IMAV 2024 Competition - Quadcopter Drone Simulation, Vision, and Control

## Project: P000377CSITCP

## NOTE:
- This is a modified version of this project without the DATASETS as they are too large to host on Github. Email me at owiemarston@gmail.com for said datasets. 

### Team Members

- Bob Qian, s3840234
- Christopher Tait, s3899475
- Martin Lawrence, s3788137
- Neil Bade, s3741453
- Owen Griffiths, s3815261

### Technical Report

**QUADCOPTER DRONE SIMULATION, VISION, AND CONTROL WITH IMAV**

## Overview

This repository contains the code and instructions to train and run a YOLOv8 model for the IMAV 2024 competition, focusing on detecting zebras and aruco markers. The training and inference are optimized for a system with a GPU, though it can be run on CPU as well (with significantly longer training times).

## Setup Instructions

### Prerequisites

- Conda (Anaconda/Miniconda)
- CUDA-compatible GPU (optional but recommended for training)
- Windows Operating System (This will change your PyTorch installation commands)

### Creating the Conda Environment

1.  **Install Conda**: Follow the instructions on the [official Conda website](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html) to install Conda on your system.

2.  **Create and Activate the Conda Environment**:

    ```bash
    conda create -n imav2024 python=3.10
    conda activate imav2024
    ```

3.  **Install Required Packages - PyTorch & Ultralytics**:
    These are the main packages for our repositiory to function - Ultralytics: A library that facilitates the use of YOLOv8 for object detection, handling training and inference. [Official Ultralytics website](https://docs.ultralytics.com/quickstart/#install-ultralytics) - PyTorch: A deep learning library utilized by Ultralytics. [Official PyTorch website](https://pytorch.org/get-started/locally/)

        ```bash
        conda install -c pytorch -c nvidia -c conda-forge pytorch torchvision pytorch-cuda=11.8 ultralytics
        ```

4.  **Troubleshooting**:
    Sometimes there is issues with the above command, if you are experiencing this, try to install each separately! 1. PyTorch
    `conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia` 2. Ultralytics
    `conda  install  -c  conda-forge  ultralytics`

## Training the YOLOv8 Model

The `model_train.py` script trains the YOLOv8 model using a specified dataset.

- **Prepare Your Dataset**: Ensure your dataset is organized and labeled correctly according to the YOLO format. Place your `data.yaml` file in the dataset directory.

- **There are provided Datasets**: These were created through [RoboFlow Annotate](https://roboflow.com/annotate) you can upload the provided datasets or create your own through RoboFlow

  - Datasets\IMAV.v4i.yolov8.zip
  - Datasets\IMAV.v7i.yolov8.zip
  - Datasets\Zebruco.v1i.yolov8.zip
  - Datasets\Zebruco.v5-zebra-zebra_single-no-aruco.yolov8.zip
  - Datasets\Zebruco.v6-zebra-only.yolov8.zip

- **Run the Training Script**:

  ```bash
  python model_train.py <path_to_your_data.yaml> <name_of_to_be_trained_model>
  ```

  Examples:

  ```bash
  python model_train.py Datasets\IMAV.v4i.yolov8.zip IMAV_v4i
  python model_train.py Datasets\IMAV.v7i.yolov8.zip IMAV_v7i
  ```

- After training has been complete, the results of the training will be placed into
  - `runs/detect/name_of_to_be_trained_model_results`
  - These results will be serperated across two folders, results and results2
  - The `weights of the trained model` will be placed into the first folder
    - Example: `runs\detect\IMAV_v4i_results\weights`
  **NOTE: THE TRAINED WEIGHTS ARE EXTREMELY IMPORTANT FOR THE BELOW SCRIPTS TO OPERATE!**

## Running Inference

The `run_inference.py` script runs inference on a directory of images using a trained YOLOv8 model and saves the prediction results. The inference is run on images.

1. **Run the Inference Script**:

   ```bash
   python run_inference.py <path_to_your_model.pt> <path_to_your_images_directory> <path_to_output_directory>
   ```

   Example:

   ```bash
   python run_inference.py runs\detect\IMAV_v4i_results\weights\best.pt images\Zebruco.v7-test-images.yolov8\test IMAV_v4i_model_predictions
   ```

## Running Video Detection

The `video_detection.py` script runs inference on a video file (.mp4) using a trained YOLOv8 model and saves the annotated video to the root of the repository.

1. **Run the Video Detection Script**:

   ```bash
   python video_detection.py <path_to_video_file> <path_to_model_file>
   ```

   Example:

   ```bash
   python video_detection.py drone-videos\DJI_0234.MP4 runs\detect\IMAV_v4i_results\weights\best.pt images\Zebruco.v7-test-images.yolov8\test
   ```

## Installing to Raspberry Pi

Once you have trained a model and now have access to `model_weights.pt`which are stored in `runs\detect\..` you can now proceed to deploy the trained model onto a Raspberry Pi

We unfortunatly have not automated this for you - this may be a work in progress!

Please follow [Ultralytics - Raspberry Pi Quick Start Guide](https://docs.ultralytics.com/guides/raspberry-pi/)

This will guide you through the steps of setting up Ultralytics on a Raspberry Pi, once this is done you can clone the repo and use the `model_weights` and accompanying scripts or feel free to just use the weights. As training on a Raspberry Pi is not reccomended!

## Authors

- Bob Qian, s3840234
- Christopher Tait, s3899475
- Martin Lawrence, s3788137
- Neil Bade, s3741453
- Owen Griffiths, s3815261

## License

This project is licensed under the MIT License.

## Acknowledgments

This project is a part of the IMAV 2024 competition, focusing on the development of computer vision models for drone simulations and control.
