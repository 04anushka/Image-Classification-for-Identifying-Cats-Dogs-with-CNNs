# Image Classification for Identifying Cats and Dogs with CNNs

## Overview

This project implements a Convolutional Neural Network (CNN) for binary image classification using TensorFlow and Keras. The goal is to classify images as either cats or dogs based on features learned by the model.

## Features

- **Data Handling:** Loads and preprocesses image data from CSV files.
- **Model Architecture:** Utilizes CNN layers (Conv2D, MaxPooling2D), Dense layers, and ReLU/sigmoid activation functions.
- **Training & Evaluation:** Trains the model and evaluates its performance with accuracy and loss metrics.
- **Visualization:** Displays sample images and model predictions.

## Installation

To get started, clone this repository and install the necessary dependencies:

```bash
git clone https://github.com/your-username/your-repository.git
cd your-repository
pip install -r requirements.txt
```

## Usage

1. **Prepare Data:** Ensure that your image data is in CSV format and formatted correctly.
2. **Run the Code:**
   ```python
   python main.py
   ```
   This will load the data, preprocess it, train the CNN, and evaluate the performance.

## Data

- `input.csv` and `input_test.csv`: Contains image pixel data.
- `labels.csv` and `labels_test.csv`: Contains corresponding labels (0 for Dog, 1 for Cat).

## Model Architecture

- **Conv2D Layers:** Extract features from images.
- **MaxPooling2D Layers:** Reduce dimensionality.
- **Flatten Layer:** Convert 2D feature maps to 1D.
- **Dense Layers:** Perform classification.
- **Activation Functions:** ReLU (hidden layers), Sigmoid (output layer).

## Results

The model achieves an accuracy of approximately X% on the test set. Further improvements can be made by tuning hyperparameters, expanding the dataset, or experimenting with more complex architectures.

## Contributing

Feel free to submit issues or pull requests. Contributions are welcome!
