"""
Traffic Sign Recognition - Version 1: Basic CNN
LeNet-inspired simple architecture

Architecture Overview:
===========================================================================
        Input (30×30×3)
            ↓
        ┌─────────────────────────────────┐
        │ Convolutional Layer             │
        ├─────────────────────────────────┤
        │ Conv2D(32, 3×3) + ReLU          │ (28×28×32)
        │ MaxPooling2D(2×2)               │ (14×14×32)
        └─────────────────────────────────┘
            ↓
        ┌─────────────────────────────────┐
        │ Flatten                         │
        ├─────────────────────────────────┤
        │ (14×14×32) → (6272)             │
        └─────────────────────────────────┘
            ↓
        ┌─────────────────────────────────┐
        │ Fully Connected Layers          │
        ├─────────────────────────────────┤
        │ Dense(128) + ReLU               │
        │ Dropout(0.5)                    │
        │ Dense(43) + Softmax             │
        └─────────────────────────────────┘
            ↓
        Output: 43 classes

Features:
- Single convolutional layer
- Simple and fast
- ~40K parameters
- Good for quick prototyping
- Expected accuracy: 70-80%

===========================================================================
"""

import cv2
import numpy as np
import os
import sys
import tensorflow as tf

from sklearn.model_selection import train_test_split
# Add parent directory to path to import utils
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import load_data

EPOCHS = 10
IMG_WIDTH = 30
IMG_HEIGHT = 30
NUM_CATEGORIES = 43
TEST_SIZE = 0.4


def main():

    # Check command-line arguments
    if len(sys.argv) not in [2, 3]:
        sys.exit("Usage: python traffic.py data_directory [model.h5]")

    # Get image arrays and labels for all image files
    print("Loading data...")
    images, labels = load_data(sys.argv[1])
    print(f"✓ Loaded {len(images)} images from {len(set(labels))} categories")

    # Split data into training and testing sets
    print("Preparing data...")
    
    # Normalization: Convert pixel values from 0-255 to 0-1
    images = np.array(images) / 255.0
    
    labels = tf.keras.utils.to_categorical(labels)
    x_train, x_test, y_train, y_test = train_test_split(
        images, np.array(labels), test_size=TEST_SIZE
    )
    print(f"✓ Training set: {len(x_train)} images")
    print(f"✓ Testing set: {len(x_test)} images")

    # Get a compiled neural network
    print("\nBuilding model...")
    model = get_model()
    print("✓ Model created\n")

    # Fit model on training data
    print("Training model...")
    model.fit(x_train, y_train, epochs=EPOCHS)

    # Evaluate neural network performance
    print("\nEvaluating model...")
    model.evaluate(x_test,  y_test, verbose=2)

    # Save model to file
    if len(sys.argv) == 3:
        filename = sys.argv[2]
        model.save(filename)
        print(f"\n✓ Model saved to {filename}.")
    
    print("\n✓ Training complete!")


def get_model():
    """
    Returns a compiled convolutional neural network model. Assume that the
    `input_shape` of the first layer is `(IMG_WIDTH, IMG_HEIGHT, 3)`.
    The output layer should have `NUM_CATEGORIES` units, one for each category.
    """
    
    
    # ============================= Version 1 ===============================
    # Create a CNN 
    model = tf.keras.models.Sequential([
        
        # Convolutional layer 
        tf.keras.layers.Conv2D(
            32, (3, 3), activation="relu", input_shape=(IMG_WIDTH, IMG_HEIGHT, 3)
        ),
        # Max-pool layer, using 2x2 pool size 
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        
        # Flatten units
        tf.keras.layers.Flatten(),
        
        # Add a hidden layer with drop out
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dropout(0.5),
        
        # Add an output layer 
        tf.keras.layers.Dense(NUM_CATEGORIES, activation="softmax")
    ])
    # =======================================================================
      
    # Compile the CNN model
    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    
    return model
    

if __name__ == "__main__":
    main()