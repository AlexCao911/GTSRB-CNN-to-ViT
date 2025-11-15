"""

Traffic Sign Recognition - Version 3: VGG Style
VGG Features:
- Uses small convolution kernels (3x3)
- Multiple convolutional blocks, each with multiple convolutional layers
- Number of filters doubles layer by layer: 32 -> 64 -> 128
- padding='same' to maintain feature map size
- Deep network with many parameters

===========================================================================
        Input (30×30×3)
            ↓
        ┌─────────────────────────────────┐
        │ Block 1: 32 filters             │
        ├─────────────────────────────────┤
        │ Conv2D(32, 3×3) + ReLU          │ (30×30×32)
        │ BatchNorm                       │
        │ Conv2D(32, 3×3) + ReLU          │ (30×30×32)
        │ BatchNorm                       │
        │ MaxPool(2×2)                    │ (15×15×32)
        │ Dropout(0.25)                   │
        └─────────────────────────────────┘
            ↓
        ┌─────────────────────────────────┐
        │ Block 2: 64 filters             │
        ├─────────────────────────────────┤
        │ Conv2D(64, 3×3) + ReLU          │ (15×15×64)
        │ BatchNorm                       │
        │ Conv2D(64, 3×3) + ReLU          │ (15×15×64)
        │ BatchNorm                       │
        │ MaxPool(2×2)                    │ (7×7×64)
        │ Dropout(0.25)                   │
        └─────────────────────────────────┘
            ↓
        ┌─────────────────────────────────┐
        │ Block 3: 128 filters            │
        ├─────────────────────────────────┤
        │ Conv2D(128, 3×3) + ReLU         │ (7×7×128)
        │ BatchNorm                       │
        │ Conv2D(128, 3×3) + ReLU         │ (7×7×128)
        │ BatchNorm                       │
        │ MaxPool(2×2)                    │ (3×3×128)
        │ Dropout(0.25)                   │
        └─────────────────────────────────┘
            ↓
        ┌─────────────────────────────────┐
        │ Fully Connected Layers          │
        ├─────────────────────────────────┤
        │ Flatten                         │ (1152)
        │ Dense(512) + ReLU               │
        │ BatchNorm + Dropout(0.5)        │
        │ Dense(256) + ReLU               │ ← VGG featurs: multiple Dense layers
        │ BatchNorm + Dropout(0.5)        │
        │ Dense(43) + Softmax             │
        └─────────────────────────────────┘

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

EPOCHS = 15  # VGG requires more training epochs
IMG_WIDTH = 30
IMG_HEIGHT = 30
NUM_CATEGORIES = 43
TEST_SIZE = 0.4


def main():

    # Check command-line arguments
    if len(sys.argv) not in [2, 3]:
        sys.exit("Usage: python traffic_v3_vgg.py data_directory [model.h5]")

    # Get image arrays and labels for all image files
    print("=" * 70)
    print("Traffic Sign Recognition - VGG Style (Version 3)")
    print("=" * 70)
    print("\nLoading data...")
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
    print("\nBuilding VGG-style model...")
    model = get_model()
    model.summary()
    print("✓ Model created\n")

    # Fit model on training data
    print("Training model...")
    history = model.fit(x_train, y_train, epochs=EPOCHS, validation_split=0.2, verbose=1)

    # Evaluate neural network performance
    print("\nEvaluating model...")
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
    print(f"\n✓ Test Accuracy: {test_acc:.4f}")
    print(f"✓ Test Loss: {test_loss:.4f}")

    # Save model to file
    if len(sys.argv) == 3:
        filename = sys.argv[2]
        model.save(filename)
        print(f"\n✓ Model saved to {filename}.")
    
    print("\n✓ Training complete!")


def get_model():
    """
    VGG-Style Architecture
    
    Features:
    - 3 convolutional blocks, each with 2 convolutional layers
    - Filters: 32 -> 64 -> 128
    - padding='same' to maintain feature map size
    - Uses Batch Normalization to speed up training
    - Deeper network structure
    """
    
    model = tf.keras.models.Sequential([
        
        # ============ Block 1: 32 filters ============ 
        tf.keras.layers.Conv2D(
            32, (3, 3), activation="relu", padding="same",
            input_shape=(IMG_WIDTH, IMG_HEIGHT, 3)
        ),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Dropout(0.25),
        
        # ============ Block 2: 64 filters ============ 
        tf.keras.layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Dropout(0.25),
        
        # ============ Block 3: 128 filters ============ 
        tf.keras.layers.Conv2D(128, (3, 3), activation="relu", padding="same"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2D(128, (3, 3), activation="relu", padding="same"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Dropout(0.25),
        
        # ============ Fully Connected Layers ============ 
        tf.keras.layers.Flatten(),
        
        # First dense layer
        tf.keras.layers.Dense(512, activation="relu"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.5),
        
        # Second dense layer (VGG feature: multiple Dense layers)
        tf.keras.layers.Dense(256, activation="relu"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.5),
        
        # Output layer
        tf.keras.layers.Dense(NUM_CATEGORIES, activation="softmax")
    ])
    
    # Compile the model
    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    
    return model


if __name__ == "__main__":
    main()