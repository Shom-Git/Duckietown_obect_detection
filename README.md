Duckiebot Object Detection Project
==================================

Overview
--------
This repository contains an object detection implementation designed specifically for a Duckiebot. The model replicates the approach of leading object detection frameworks such as YOLO, solving an optimization problem that minimizes a combination of localization and classification losses during training.

Training Procedure
------------------
The training was conducted on Google Cloud using GPUs, with resources obtained as part of a Kaggle-hosted competition. Thanks to the $1000 cloud credit awarded from the tournament, significant training capacity was available for experimenting and optimizing the model. The training process involved supervised learning on annotated datasets, where the goal was to predict bounding box coordinates and object classes simultaneously. This was achieved using a convolutional neural network trained to minimize loss over predicted vs. actual positions and class probabilities.

Deployment
----------
After training, the model was deployed on a Duckiebot system. The deployment mechanism was built using ROS (Robot Operating System) to ensure real-time inference and communication with the Duckiebot's hardware. The node structure included:
- **Subscriber** to the onboard camera feed.
- **Publisher** that sends annotated frames (with bounding boxes) to the display or screen topic.

ROS Integration
---------------
For system integration and deployment, the project relies on the official Duckietown ROS infrastructure. Specifically, the `template-ros` package was used as the foundation for the ROS workspace and node architecture. You can find it here:
https://github.com/duckietown/template-ros

Key Features
------------
- Real-time object detection on Duckiebot camera feed.
- ROS-based architecture with minimal latency.
- Optimized training via GPU-enabled cloud infrastructure.
- End-to-end pipeline: from dataset to deployment on physical robot.

Note
----
This implementation does not rely on external detection frameworks (like YOLOv5) directly but instead recreates a similar logic and objective through custom model design and training flow.
