Convolutional Neural Network (CNN) Implementation in NumPy
Overview
This project implements a Convolutional Neural Network (CNN) from scratch using NumPy for classifying handwritten digits from the MNIST dataset. The implementation avoids high-level frameworks like TensorFlow or PyTorch, focusing on a manual approach to understand the core operations of a CNN, including convolution, forward/backward propagation, and gradient descent.
Features

Data Loading and Preprocessing: Loads and normalizes the MNIST dataset.
Model Architecture: Implements a simple feedforward neural network with one hidden layer (not a true CNN with convolutional layers, but serves as a baseline for classification).
Training: Uses gradient descent with backpropagation to optimize model parameters.
Prediction and Evaluation: Provides functions to predict digit classes and compute accuracy.
Visualization: Displays sample predictions with corresponding images.

Requirements

Python 3.x
NumPy
Pandas
Matplotlib

Install dependencies using:
pip install numpy pandas matplotlib

Dataset
The project uses the MNIST dataset (provided as mnist_train.csv in the project directory). The dataset contains 60,000 training images of handwritten digits (28x28 pixels) with corresponding labels (0-9).
Project Structure

cnn_model.ipynb: Jupyter Notebook containing the full implementation, including data preprocessing, model architecture, training, and evaluation.
mnist_train.csv: MNIST dataset file (ensure it is placed in the correct directory as specified in the notebook).
README.md: This file.

Usage

Clone the Repository:
git clone <repository-url>
cd <repository-directory>


Ensure Dataset Availability: Place the mnist_train.csv file in the directory specified in the notebook (e.g., C:\Users\PC\Documents\Convolutional Neural Network(CNN)\).

Run the Notebook: Open cnn_model.ipynb in Jupyter Notebook or JupyterLab and execute the cells sequentially:
jupyter notebook cnn_model.ipynb


Key Steps in the Notebook:

Data Preprocessing: Loads and normalizes the MNIST dataset.
Model Initialization: Initializes weights and biases for the neural network.
Training: Trains the model using gradient descent for 1000 iterations.
Evaluation: Tests predictions on sample images and visualizes results.



Model Architecture
The model is a simple feedforward neural network (not a true CNN due to the absence of convolutional layers):

Input Layer: Flattened 28x28 images (784 neurons).
Hidden Layer: 10 neurons with ReLU activation.
Output Layer: 10 neurons (one per digit) with Softmax activation.
Loss Function: Cross-Entropy Loss.
Optimizer: Gradient Descent.

Results

The model achieves an accuracy of approximately 90.94% on the training set after 1000 iterations (as shown in the notebook output).
Sample predictions are visualized, showing the predicted digit, true label, and the corresponding image.

