# Image Classification using CNN on CIFAR-10

## Overview

This project implements a Convolutional Neural Network (CNN) for image classification using the CIFAR-10 dataset. The model is trained to classify images into 10 categories and achieves an accuracy of 77.21%.

## Dataset

CIFAR-10 contains 60,000 color images (32×32 pixels) belonging to the following classes:

* Airplane
* Automobile
* Bird
* Cat
* Deer
* Dog
* Frog
* Horse
* Ship
* Truck

Dataset Split:

* Training Images: 50,000
* Testing Images: 10,000

## Technologies Used

* Python
* PyTorch
* Torchvision
* NumPy
* Matplotlib
* Scikit-learn

## CNN Architecture

* Conv2D (32 Filters)
* ReLU
* Max Pooling
* Conv2D (64 Filters)
* ReLU
* Max Pooling
* Fully Connected Layer (512 Neurons)
* Dropout
* Output Layer (10 Classes)

## Training Configuration

* Optimizer: Adam
* Learning Rate: 0.001
* Epochs: 20
* Batch Size: 64
* Loss Function: CrossEntropyLoss

## Performance

* Accuracy: 77.21%
* Precision: 77%
* Recall: 77%
* F1-Score: 77%

## Results

Generated Outputs:

* Training Loss Curve
* Confusion Matrix
* Cat and Dog Sample Images
* Trained CNN Model

## Project Structure

ComputerVision_CNN_Project/

* model.py
* train.py
* evaluate.py
* show_samples.py
* requirements.txt
* README.md

results/

* loss_curve.png
* confusion_matrix.png
* cat_dog_samples.png

saved_models/

* cnn_cifar10.pth

## How to Run

Install dependencies:

pip install -r requirements.txt

Train the model:

python train.py

Evaluate the model:

python evaluate.py

Display sample images:

python show_samples.py

## Author

Muhammad Aman Fasihe

BS Artificial Intelligence

The Islamia University of Bahawalpur
Note: The trained model file (.pth) is included in the project submission package. The GitHub repository contains the source code, documentation, and result visualizations.

