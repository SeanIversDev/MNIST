# MNIST Digit Classifier

A convolutional neural network trained on the MNIST dataset to classify handwritten digits (0–9), with a Streamlit web app for real-time inference.

## Overview

The model was trained using TensorFlow with GPU acceleration via WSL (Windows Subsystem for Linux), achieving **97.7% accuracy** on the test set after 5 epochs.

## Model Architecture

| Layer | Details |
|-------|---------|
| Input | 28x28 grayscale image |
| Flatten | Converts 2D image to 1D vector |
| Dense | 128 units, ReLU activation |
| Dense | 10 units, Softmax activation (one per digit) |

**Optimizer:** Adam  
**Loss:** Sparse Categorical Crossentropy

## Training Results

| Epoch | Accuracy | Loss |
|-------|----------|------|
| 1 | 92.6% | 0.2599 |
| 2 | 96.6% | 0.1144 |
| 3 | 97.7% | 0.0777 |
| 4 | 98.2% | 0.0582 |
| 5 | 98.6% | 0.0452 |

**Test Accuracy: 97.7%**

## Project Structure

```
MNIST/
├── MNIST.ipynb        # Model training notebook
├── MNIST.py           # Streamlit inference app
└── mnist_model.keras  # Saved trained model
```

## Running the App

```bash
streamlit run MNIST.py
```

Upload a 28x28 grayscale image of a handwritten digit and the app will predict which digit (0–9) it is.

## Environment

- **Platform:** WSL (Windows Subsystem for Linux)
- **GPU acceleration:** TensorFlow GPU via WSL
- **Python libraries:** TensorFlow, Streamlit, NumPy, Pillow

## Dataset

The [MNIST dataset](http://yann.lecun.com/exdb/mnist/) consists of 70,000 handwritten digit images (60,000 training, 10,000 test), each 28x28 pixels in grayscale.
