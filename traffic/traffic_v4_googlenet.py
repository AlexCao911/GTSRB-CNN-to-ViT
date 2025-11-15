"""
Traffic Sign Recognition - Version 4: GoogLeNet (Inception) Style
GoogLeNet Features:
- Inception module: Uses convolutional kernels of different sizes in parallel
- 1x1 convolution for dimensionality reduction
- Multi-scale feature extraction
- High parameter efficiency (fewer than VGG)

===========================================================================
        Input (30×30×3)
            ↓
        Conv2D(32, 3×3) + BN → MaxPool(2×2)  (15×15×32)
            ↓
        ┌─────────────────────────────────────────────┐
        │ Inception Module 1                          │
        ├─────────────────────────────────────────────┤
        │ Branch 1: 1×1(16)                           │
        │ Branch 2: 1×1(16) → 3×3(32)                 │
        │ Branch 3: 1×1(8) → 5×5(16)                  │
        │ Branch 4: MaxPool → 1×1(8)                  │
        │ Concatenate: 16+32+16+8 = 72 filters        │
        └─────────────────────────────────────────────┘
            ↓ (15×15×72)
        BatchNorm + Dropout(0.25)
            ↓
        MaxPool(2×2)  (7×7×72)
            ↓
        ┌─────────────────────────────────────────────┐
        │ Inception Module 2                          │
        ├─────────────────────────────────────────────┤
        │ Branch 1: 1×1(32)                           │
        │ Branch 2: 1×1(32) → 3×3(64)                 │
        │ Branch 3: 1×1(16) → 5×5(32)                 │
        │ Branch 4: MaxPool → 1×1(16)                 │
        │ Concatenate: 32+64+32+16 = 144 filters      │
        └─────────────────────────────────────────────┘
            ↓ (7×7×144)
        BatchNorm + Dropout(0.25)
            ↓
        ┌─────────────────────────────────────────────┐
        │ Global Average Pooling                      │ ← GoogLeNet features
        ├─────────────────────────────────────────────┤
        │ Turn (7×7×144) into (144)                   │
        │ Each channels get its average value         │
        │ Reduce params                               │
        └─────────────────────────────────────────────┘
            ↓ (144)
        Dense(512) + BN + Dropout(0.5)
            ↓
        Dense(43) + Softmax

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

EPOCHS = 15
IMG_WIDTH = 30
IMG_HEIGHT = 30
NUM_CATEGORIES = 43
TEST_SIZE = 0.4


def main():

    # Check command-line arguments
    if len(sys.argv) not in [2, 3]:
        sys.exit("Usage: python traffic_v4_googlenet.py data_directory [model.h5]")

    # Get image arrays and labels for all image files
    print("=" * 70)
    print("Traffic Sign Recognition - GoogLeNet/Inception Style (Version 4)")
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
    print("\nBuilding GoogLeNet-style model...")
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


def inception_module(x, filters_1x1, filters_3x3_reduce, filters_3x3, 
                     filters_5x5_reduce, filters_5x5, filters_pool_proj):
    """
    Inception Module
    
    Performs 4 operations in parallel:
    1. 1x1 convolution
    2. 1x1 convolution -> 3x3 convolution
    3. 1x1 convolution -> 5x5 convolution
    4. 3x3 MaxPooling -> 1x1 convolution
    
    Then concatenates the results along the channel dimension
    """
    
    # Branch 1: 1x1 convolution
    branch1 = tf.keras.layers.Conv2D(filters_1x1, (1, 1), padding='same', activation='relu')(x)
    
    # Branch 2: 1x1 convolution -> 3x3 convolution
    branch2 = tf.keras.layers.Conv2D(filters_3x3_reduce, (1, 1), padding='same', activation='relu')(x)
    branch2 = tf.keras.layers.Conv2D(filters_3x3, (3, 3), padding='same', activation='relu')(branch2)
    
    # Branch 3: 1x1 convolution -> 5x5 convolution
    branch3 = tf.keras.layers.Conv2D(filters_5x5_reduce, (1, 1), padding='same', activation='relu')(x)
    branch3 = tf.keras.layers.Conv2D(filters_5x5, (5, 5), padding='same', activation='relu')(branch3)
    
    # Branch 4: 3x3 MaxPooling -> 1x1 convolution
    branch4 = tf.keras.layers.MaxPooling2D((3, 3), strides=(1, 1), padding='same')(x)
    branch4 = tf.keras.layers.Conv2D(filters_pool_proj, (1, 1), padding='same', activation='relu')(branch4)
    
    # Concatenate all branches
    output = tf.keras.layers.concatenate([branch1, branch2, branch3, branch4], axis=-1)
    
    return output


def get_model():
    """
    GoogLeNet/Inception-Style Architecture
    
    Features:
    - Uses Inception modules for multi-scale feature extraction
    - 1x1 convolution for dimensionality reduction to reduce parameters
    - Parallel processing to extract features at different scales
    - High parameter efficiency
    """
    
    # Use Functional API (because the Inception module requires a branching structure)
    inputs = tf.keras.Input(shape=(IMG_WIDTH, IMG_HEIGHT, 3))
    
    # ============ Initial Convolutional Layer ============
    x = tf.keras.layers.Conv2D(32, (3, 3), padding='same', activation='relu')(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling2D((2, 2))(x)
    
    # ============ Inception Module 1 ============
    # Parameters: (1x1, 3x3_reduce, 3x3, 5x5_reduce, 5x5, pool_proj)
    x = inception_module(x, 16, 16, 32, 8, 16, 8)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.25)(x)
    
    # ============ Dimensionality Reduction ============
    x = tf.keras.layers.MaxPooling2D((2, 2))(x)
    
    # ============ Inception Module 2 ============
    x = inception_module(x, 32, 32, 64, 16, 32, 16)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.25)(x)
    
    # ============ Global Average Pooling (GoogLeNet feature) ============
    # Use Global Average Pooling instead of Flatten to reduce parameters
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    
    # ============ Fully Connected Layers ============
    x = tf.keras.layers.Dense(512, activation='relu')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    
    # ============ Output Layer ============
    outputs = tf.keras.layers.Dense(NUM_CATEGORIES, activation='softmax')(x)
    
    # Create model
    model = tf.keras.Model(inputs=inputs, outputs=outputs, name='GoogLeNet_Style')
    
    # Compile the model
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model


if __name__ == "__main__":
    main()