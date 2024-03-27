# Traffic Sign Detection with PyTorch

Welcome to the Traffic Sign Detection project! This repository contains a deep learning model designed to detect and classify traffic signs in images or real-time footage. Utilizing the power of Convolutional Neural Networks (CNN) and PyTorch, our model aims to provide high accuracy in identifying various traffic signs, making it a valuable tool for autonomous driving systems and traffic monitoring.


## Project Overview
This repository is currently under development. It aims to achieve high accuracy from a relatively small dataset.
The Traffic Sign Detection project leverages a CNN model implemented in PyTorch to recognize traffic signs from static images or video streams. The model is trained on a comprehensive dataset of traffic signs, enabling it to detect and classify a wide range of sign types under different environmental conditions.

## Dataset Information

The dataset for this project was sourced from the Big Data and Artificial Intelligence Laboratory at Fırat University. It features a collection of traffic sign images across 39 distinct classes. To ensure uniformity, all images and their corresponding labels have been resized and normalized. This careful preparation facilitates the efficient training and evaluation of our Convolutional Neural Network (CNN) model for traffic sign detection.

For more details and to download the dataset, visit the [Fırat University Big Data and AI Lab datasets page](http://buyukveri.firat.edu.tr/veri-setleri/).



## Features


- High accuracy CNN model implemented in PyTorch.
- Utilization of torchvision transforms for image preprocessing.
- Custom dataset and DataLoader for efficient training and validation.
- Evaluation metrics for model performance assessment.

## Prerequisites

Before you can run the project, you'll need to install the following software:

- Python 3.x
- PyTorch
- torchvision
- PIL (Python Imaging Library)
- pandas

## Installation

Clone this repository to your local machine:

```bash
git clone https://github.com/mr-ozdemir/Traffic_sign_detection_model_with_CNN
```
```bash
cd Traffic_sign_detection_model_with_CNN
```
```bash
pip install requirements.txt
```
