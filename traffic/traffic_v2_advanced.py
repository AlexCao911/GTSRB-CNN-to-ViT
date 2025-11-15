"""
Traffic Sign Recognition - Version 2: Improved CNN
AlexNet-inspired architecture with dual convolutional blocks

Architecture Overview:
===========================================================================
        Input (30×30×3)
            ↓
        ┌─────────────────────────────────┐
        │ Convolutional Block 1           │
        ├─────────────────────────────────┤
        │ Conv2D(32, 3×3) + ReLU          │ (28×28×32)
        │ Conv2D(32, 3×3) + ReLU          │ (26×26×32)
        │ MaxPooling2D(2×2)               │ (13×13×32)
        │ Dropout(0.25)                   │
        └─────────────────────────────────┘
            ↓
        ┌─────────────────────────────────┐
        │ Convolutional Block 2           │
        ├─────────────────────────────────┤
        │ Conv2D(64, 3×3) + ReLU          │ (11×11×64)
        │ Conv2D(64, 3×3) + ReLU          │ (9×9×64)
        │ MaxPooling2D(2×2)               │ (4×4×64)
        │ Dropout(0.25)                   │
        └─────────────────────────────────┘
            ↓
        ┌─────────────────────────────────┐
        │ Flatten                         │
        ├─────────────────────────────────┤
        │ (4×4×64) → (1024)               │
        └─────────────────────────────────┘
            ↓
        ┌─────────────────────────────────┐
        │ Fully Connected Layers          │
        ├─────────────────────────────────┤
        │ Dense(512) + ReLU               │
        │ Dropout(0.5)                    │
        │ Dense(43) + Softmax             │
        └─────────────────────────────────┘
            ↓
        Output: 43 classes

Features:
- Two convolutional blocks
- Dual convolutions per block (deeper feature extraction)
- Filter progression: 32 → 64
- Dropout for regularization
- ~612K parameters
- Expected accuracy: 90-93%
- Best balance of performance and efficiency

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
    
    # ============================= Version 2 ===============================
    # Create a CNN with improved architecture
    model = tf.keras.models.Sequential([
        
        # First convolutional block
        tf.keras.layers.Conv2D(
            32, (3, 3), activation="relu", input_shape=(IMG_WIDTH, IMG_HEIGHT, 3)
        ),
        tf.keras.layers.Conv2D(32, (3, 3), activation="relu"),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Dropout(0.25),
        
        # Second convolutional block
        tf.keras.layers.Conv2D(64, (3, 3), activation="relu"),
        tf.keras.layers.Conv2D(64, (3, 3), activation="relu"),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Dropout(0.25),
        
        # Flatten and dense layers
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(512, activation="relu"),
        tf.keras.layers.Dropout(0.5),
        
        # Output layer
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