Duckiebot Object Detection Project
==================================

Overview
--------
This repository contains an object detection implementation designed specifically for a Duckiebot. The model replicates the approach of leading object detection frameworks such as YOLO, solving an optimization problem that minimizes a combination of localization and classification losses during training.

YOLO-Inspired Optimization Objective
------------------------------------
The model is trained to detect objects by solving a joint optimization problem. This includes:

1. **Localization Loss** — Measures the error in predicted bounding box coordinates (x, y, width, height).
2. **Confidence Loss** — Measures how confidently the model believes there is an object in a predicted box.
3. **Classification Loss** — Measures the accuracy of the predicted class of the detected object.

Mathematically, the model solves:

    min_θ ∑[i=1 to N] 𝓛(ŷᵢ, yᵢ; θ)

Where:
- θ are the model parameters (weights),
- yᵢ is the ground truth for the i-th image,
- ŷᵢ is the model’s prediction,
- 𝓛 is the total loss function combining the above components.

The optimizer used was Adam, iteratively, component wise adjusting weights, with momentum to minimize this objective over training data.

Training Procedure
------------------
The training was conducted on Google Cloud using GPUs. Access was granted via a Kaggle-hosted tournament with $1000 in cloud credit, which provided ample compute resources for model experimentation. Training was supervised, learning to predict bounding boxes and class labels from annotated data.

Dataset
-------
The dataset was initially obtained from the official Duckietown object detection repository:
https://github.com/duckietown/duckietown-objdet

Due to incompatibilities with the model input format, the dataset was manually reconstructed. This ensured proper alignment with the model's expected input dimensions and annotation structure.

Deployment
----------
After training, the model was deployed on a Duckiebot using ROS (Robot Operating System). The deployment consisted of:
- A **subscriber** node to receive camera feed,
- A **publisher** node to output annotated images to a screen topic.

ROS Integration
---------------
This project integrates with the official Duckietown ROS infrastructure, using the `template-ros` package as its foundation. It provides a standard environment for building and deploying ROS-based robotics projects:
https://github.com/duckietown/template-ros

Key Features
------------
- Real-time object detection using onboard Duckiebot camera.
- ROS-based deployment for reliable robotic system integration.
- Trained with GPU acceleration on Google Cloud using competition credit.
- Dataset manually adapted to match model input.
- End-to-end pipeline: from raw data to deployed model.

Note
----
This project does not use YOLOv5 directly, but reconstructs the core ideas and training objectives that YOLO-like models optimize for.

About the Project
-----------------
Module Name: S25_CO-548-A_RIS Project  
Instructor: Ph.D Kristina Nikolovska
