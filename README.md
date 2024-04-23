# Provice Vision AI Examples

This repository contains example code and workflows for building, training, testing, and evaluating models for various image analysis tasks. Each task is accompanied by its own set of example code in Google Colab notebooks.

## Task Overview

- **Binary Classification**: Classify images into two categories.
- **Multi-Class Classification**: Classify images into multiple categories.
- **Multi-Label Classification**: Assign multiple labels to each image.
- **Image Retrieval**: Retrieve images based on similarity.
- **Object Detection**: Detect and localize objects within images.
- **Instance Segmentation**: Segment and classify each instance of objects within images.

## Dataset Construction

- **Binary Classification**: Prepare a dataset with labeled images corresponding to the two classes of interest.
- **Multi-Class Classification**: Construct a dataset with labeled images covering multiple categories.
- **Multi-Label Classification**: Create a dataset with images annotated with multiple labels.
- **Image Retrieval**: Construct a dataset with images and corresponding metadata defining similarity.
- **Object Detection**: Create a dataset with images annotated with bounding boxes around objects of interest.
- **Instance Segmentation**: Prepare a dataset with images annotated with pixel-level masks for each instance of objects.

## Training

- Utilize the provided Colab notebooks to train models on the prepared datasets using popular deep learning frameworks such as PyTorch.
- Adjust hyperparameters, network architectures, and optimization strategies based on the specific task and dataset characteristics.

## Testing

- Use the trained models to perform inference on a separate test dataset.
- Evaluate the performance of the models using appropriate metrics for each task, such as accuracy, mean Average Precision (mAP), or Intersection over Union (IoU).

## Evaluation

- Analyze the results of testing to assess the effectiveness of the trained models.
- Compare the performance of different models and configurations to identify the most suitable approach for the given task and dataset.

## Computer Vision Packages

- timm: PyTorch Image Models for image classification tasks.
- mmdetection: OpenMMLab's toolbox for object detection and instance segmentation.
- yolo: You Only Look Once framework for object detection tasks.

## Repository Structure

- **`01_binary_classification/`**: Contains Colab notebooks and scripts for binary classification.
- **`02_multi_class_classification/`**: Contains Colab notebooks and scripts for multi-class classification.
- **`03_multi_label_classification/`**: Contains Colab notebooks and scripts for multi-label classification.
- **`04_object_detection/`**: Contains Colab notebooks and scripts for object detection.
- **`05_instance_segmentation/`**: Contains Colab notebooks and scripts for instance segmentation.
- **`06_image_retrieval/`**: Contains Colab notebooks and scripts for image retrieval.

## Usage

1. Clone the repository: `git clone https://github.com/kimnamu/vision_ai.git`
2. Navigate to the desired task directory.
3. Follow the instructions provided in the README of each task directory to execute the example code in Google Colab.

## Copyright
©kimnamu(jihoon.lucas.kim@gmail.com). All rights reserved.