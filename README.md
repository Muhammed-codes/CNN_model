# Convolutional Neural Network (CNN) Implementation in NumPy

## Overview

This project implements a **Convolutional Neural Network (CNN)** from scratch using **NumPy** for classifying handwritten digits from the **MNIST dataset**. The implementation avoids high-level frameworks like TensorFlow or PyTorch, focusing on a manual approach to understand the core operations of a CNN, including convolution, forward/backward propagation, and gradient descent.

## Features

1. **Data Loading and Preprocessing**: Loads and normalizes the MNIST dataset.
2. **Model Architecture**: Implements a simple feedforward neural network with one hidden layer (not a true CNN with convolutional layers, but serves as a baseline for classification).
3. **Training**: Uses gradient descent with backpropagation to optimize model parameters.
4. **Prediction and Evaluation**: Provides functions to predict digit classes and compute accuracy.
5. **Visualization**: Displays sample predictions with corresponding images.

## Requirements

- Python 3.x
- NumPy
- Pandas
- Matplotlib

Install dependencies using:

```bash
pip install numpy pandas matplotlib
