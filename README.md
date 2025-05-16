# Duckietown_obect_detection

[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Introduction to YOLO

You Only Look Once (YOLO) is a state-of-the-art, real-time object detection system. Unlike traditional object detection methods that repurpose classifiers to perform detection, YOLO frames object detection as a regression problem to spatially separated bounding boxes and associated class probabilities. A single neural network directly predicts bounding boxes and class probabilities from full images in one evaluation. This end-to-end approach allows for significantly faster inference speeds, making it ideal for real-time applications. The current state-of-the-art implementation from Ultralytics is YOLOv8.

## Setting Up Your Environment with Anaconda and Ultralytics

This section guides you through setting up a dedicated environment using Anaconda and installing the Ultralytics YOLOv8 library via pip.

1.  **Install Anaconda:** If you haven't already, download and install Anaconda from the [official Anaconda website](https://www.anaconda.com/download/). Follow the installation instructions for your operating system.

2.  **Create a New Environment:** Open your terminal or Anaconda Prompt and create a new environment. We'll call it `yolo_env`, but you can choose any name you prefer:

    ```bash
    conda create --name yolo_env python=3.10  # Or your preferred Python version
    ```

3.  **Activate the Environment:** Activate the newly created environment:

    ```bash
    conda activate yolo_env
    ```

4.  **Install Ultralytics:** With your environment activated, install the Ultralytics YOLOv8 package using pip:

    ```bash
    pip install ultralytics
    ```

    This command will download and install the necessary dependencies for working with YOLOv8.

5.  **Verify Installation (Optional):** You can verify the installation by running a simple YOLO command:

    ```bash
    yolo --version
    ```

    This should print the installed Ultralytics version.

## Customizing Your Dataset with Roboflow for Object Detection

Roboflow is a powerful platform that simplifies the process of preparing and managing datasets for computer vision tasks, including object detection. Here's a general workflow for customizing your dataset using Roboflow:

1.  **Create a Roboflow Account:** Sign up for a free account at [https://roboflow.com/](https://roboflow.com/).

2.  **Create a New Project:** Once logged in, create a new project and select "Object Detection" as the project type.

3.  **Upload Your Images and Annotations:** Upload your images to the Roboflow project. If you have existing annotations (e.g., in Pascal VOC, COCO format), you can upload them as well. Roboflow supports various annotation formats.

4.  **Annotate Your Images (if needed):** If your images are not yet annotated, Roboflow provides an intuitive web interface for drawing bounding boxes around the objects of interest and assigning labels.

5.  **Apply Preprocessing and Augmentations:** Roboflow allows you to apply various preprocessing steps (e.g., resizing, cropping) and augmentations (e.g., flipping, rotating, brightness adjustments) to your dataset. These steps can significantly improve the robustness and generalization of your trained model.

6.  **Generate Your Dataset:** Once you've annotated and processed your data, you can generate a dataset version in the YOLO format. Roboflow will automatically create the necessary files (images and corresponding `.txt` annotation files) organized in the way YOLO expects.

7.  **Download Your Dataset:** Download the generated YOLO format dataset. Roboflow provides a `data.yaml` file that contains the paths to your training, validation, and test sets, as well as the class names. This file is crucial for training your YOLOv8 model.

## Training Your Custom Model Locally

To train your custom YOLOv8 model on your local machine, you will typically use the Ultralytics CLI or Python API. Ensure you have the necessary hardware (ideally a dedicated GPU with CUDA set up) and have installed the required dependencies, including PyTorch.

1.  **Organize Your Dataset:** Place the Roboflow-downloaded dataset (the `data.yaml` file and the `train`, `val`, and `test` image and label folders) in a suitable directory on your local machine.

2.  **Initiate Training using the CLI:** Open your terminal, activate your `yolo_env` Anaconda environment, and navigate to a directory where you want to run the training command. Use the `yolo train` command, specifying the path to your `data.yaml` file, the desired model architecture, and other training parameters:

    ```bash
    yolo train data=path/to/your/data.yaml model=yolov8n.yaml epochs=100 imgsz=640
    ```

    **Replace:**
    * `path/to/your/data.yaml` with the actual path to your `data.yaml` file.
    * `yolov8n.yaml` with the desired YOLOv8 model configuration file (e.g., `yolov8s.yaml`, `yolov8m.yaml`).
    * `epochs=100` with the number of training epochs you want to run.
    * `imgsz=640` with the input image size. Adjust these parameters based on your hardware and dataset.

3.  **Monitor Training:** Ultralytics will display training progress in your terminal, including loss values, mAP scores, and other metrics. Training results, including the trained weights (`best.pt`), will typically be saved in a `runs/train/` directory.

## Running Object Detection with Your Trained Model Using the Repository Code

Once you have trained your custom YOLOv8 model and have the trained weights file (e.g., `best.pt`), follow these steps to perform object detection using the provided `yolo_detect.py` script in this repository within your created Anaconda environment:

1.  **Locate Your Trained Model:** After training, find the `.pt` weights file (e.g., `best.pt`) in the `runs/train/` directory (or a similar location depending on your Ultralytics version and training configuration). Rename this file to `my_model.pt` for consistency with the command below.

2.  **Place Your Model in the Environment (Recommended):**
    * Navigate to your activated Anaconda environment directory (usually in your Anaconda installation folder under `envs/yolo_env/`).
    * Create a new directory within this environment (e.g., `trained_models`).
    * Copy the renamed `my_model.pt` file into this `trained_models` directory.

3.  **Run the Detection Script:** From the root of your repository (make sure your Anaconda environment `yolo_env` is still activated), use the following command to run the `yolo_detect.py` script:

    ```bash
    python yolo_detect.py --model path/to/your/anaconda/envs/yolo_env/trained_models/my_model.pt --source usb0 --resolution 1280x720
    ```

    **Replace:**
    * `path/to/your/anaconda/envs/yolo_env/trained_models/my_model.pt` with the actual path to your `my_model.pt` file within the Anaconda environment (or the local path if you didn't copy it).
    * `--source usb0` assumes your webcam is accessible via `/dev/video0` (on Linux-based systems). Adjust this according to your operating system and camera setup.
    * `--resolution 1280x720` sets the desired input resolution. Ensure your camera supports this resolution.

    Refer to the comments within your `yolo_detect.py` script for other available options and configurations. Ensure that the script imports the necessary libraries (likely including `ultralytics`) and is configured to load and use the provided model path and source.

**Note:** If "YOLOv11" becomes a publicly available and supported version by Ultralytics in the future, the specific commands and configurations might differ. Please refer to the official documentation for that version when it is released.
