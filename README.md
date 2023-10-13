# TensorFlow Image Classification

## Project Overview

This project focuses on image classification using the TensorFlow framework. It involves loading and exploring the CIFAR-10 dataset, building a convolutional neural network (CNN) model for image classification, training the model, and evaluating its performance. The CIFAR-10 dataset contains 60,000 32x32 color images across 10 classes, making it a common choice for benchmarking image classification models.

## Table of Contents

- [Loading the CIFAR-10 Dataset](#loading-the-cifar-10-dataset)
- [Dataset Description](#dataset-description)
- [Exploring and Analyzing the Dataset](#exploring-and-analyzing-the-dataset)
- [Normalizing Dataset](#normalizing-dataset)
- [Creating the Machine Learning Model](#creating-the-machine-learning-model)
- [Building the Machine Learning Model](#building-the-machine-learning-model)
- [Compiling the Machine Learning Model](#compiling-the-machine-learning-model)
- [The First Batch of Dataset Training](#the-first-batch-of-dataset-training)
- [The Second Batch of Dataset Training After Data Augmentation](#the-second-batch-of-dataset-training-after-data-augmentation)
- [Evaluating the Machine Learning Model](#evaluating-the-machine-learning-model)

## Loading the CIFAR-10 Dataset

I started by loading the CIFAR-10 dataset, which contains both a training set and a test set. The dataset consists of 32x32 color images, with 10 different classes.

## Dataset Description

### CIFAR-10 Dataset

The CIFAR-10 dataset is a widely used image classification dataset. It consists of 60,000 32x32 color images in 10 different classes, with each class containing 6,000 images. This dataset is a subset of the larger CIFAR-100 dataset, focusing on 10 mutually exclusive classes. The CIFAR-10 dataset is often used for training and evaluating machine learning and deep learning models for image classification tasks.

### Dataset Classes

The CIFAR-10 dataset is divided into the following 10 classes:

1. Airplane
2. Automobile
3. Bird
4. Cat
5. Deer
6. Dog
7. Frog
8. Horse
9. Ship
10. Truck

### Dataset Split

The dataset is typically split into two sets:

- **Training Set**: The training set consists of 50,000 images (5,000 images per class) used for training machine learning models.

- **Test Set**: The test set comprises 10,000 images (1,000 images per class) used for evaluating the performance of trained models.

### Dataset Characteristics

- **Image Size**: All images in the CIFAR-10 dataset are of size 32x32 pixels.
- **Color Channels**: Images are in color, containing three color channels: red, green, and blue (RGB).
- **Labeling**: Each image is labeled with one of the 10 class labels.

## Exploring and Analyzing the Dataset

I conducted an initial exploration of the dataset, including visualizing sample images, plotting class distributions, and normalizing pixel values. This provided a better understanding of the data before model building.

## Creating the Machine Learning Model

I designed a convolutional neural network (CNN) model for image classification. The model architecture included convolutional layers, batch normalization, max-pooling layers, and fully connected layers.

## Compiling the Machine Learning Model

I compiled the model with the Adam optimizer and the sparse categorical cross-entropy loss function for training. Additionally, I tracked the accuracy metric during training to evaluate model performance.

## Training the Model

I trained the model using the CIFAR-10 training dataset. The model underwent two training phases: the first batch of training without data augmentation and the second batch of training after applying data augmentation techniques.

### The First Batch of Dataset Training

In this training phase, I trained the model using the original training dataset. The training progress was tracked over 10 epochs, and the model's accuracy and loss were recorded.

### The Second Batch of Dataset Training After Data Augmentation

For the second training phase, I applied data augmentation to the training dataset to improve the model's generalization. Data augmentation techniques included random shifts and horizontal flips. The model was trained for an additional 10 epochs, and its performance was evaluated.

## Evaluating the Machine Learning Model

I evaluated the model's performance by visualizing accuracy trends throughout training and generating a classification report. The classification report provided details on precision, recall, and F1-score for each class.

## Acknowledgments

This project is made possible through the use of TensorFlow, an open-source machine learning framework. Additionally, it leveraged the CIFAR-10 dataset for image classification, a valuable resource in the machine learning community.

---
