"""
Traffic Sign Recognition - Version 5: ResNet Style
Revolutionary residual learning architecture

Architecture Overview:
===========================================================================
        Input (30×30×3)
            ↓
        ┌─────────────────────────────────┐
        │ Initial Convolution             │
        ├─────────────────────────────────┤
        │ Conv2D(32, 3×3) + BN + ReLU     │ (30×30×32)
        └─────────────────────────────────┘
            ↓
        ┌─────────────────────────────────────────────────────────┐
        │ Stage 1: 32 filters (Residual Blocks × 2)               │
        ├─────────────────────────────────────────────────────────┤
        │ Residual Block 1:                                       │
        │   x → Conv(32) → BN → ReLU → Conv(32) → BN → (+) → ReLU │
        │   |_____________________________________________↑       │
        │                                                         │
        │ Residual Block 2: (same structure)                      │
        │ Output: (30×30×32)                                      │
        │ Dropout(0.2)                                            │
        └─────────────────────────────────────────────────────────┘
            ↓
        ┌─────────────────────────────────────────────────────────┐
        │ Stage 2: 64 filters (Residual Blocks × 2)               │
        ├─────────────────────────────────────────────────────────┤
        │ Residual Block 3 (with projection):                     │
        │   x → Conv(64, s=2) → BN → ReLU → Conv(64) → BN → (+)   │
        │   |→ Conv(64, 1×1, s=2) → BN ___________________↑       │
        │                                                         │
        │ Residual Block 4: (identity shortcut)                   │
        │ Output: (15×15×64)                                      │
        │ Dropout(0.2)                                            │
        └─────────────────────────────────────────────────────────┘
            ↓
        ┌─────────────────────────────────────────────────────────┐
        │ Stage 3: 128 filters (Residual Blocks × 2)              │
        ├─────────────────────────────────────────────────────────┤
        │ Residual Block 5 (with projection):                     │
        │   x → Conv(128, s=2) → BN → ReLU → Conv(128) → BN → (+) │
        │   |→ Conv(128, 1×1, s=2) → BN __________________↑       │
        │                                                         │
        │ Residual Block 6: (identity shortcut)                   │
        │ Output: (7×7×128)                                       │
        │ Dropout(0.3)                                            │
        └─────────────────────────────────────────────────────────┘
            ↓
        ┌─────────────────────────────────┐
        │ Global Average Pooling          │
        ├─────────────────────────────────┤
        │ (7×7×128) → (128)               │
        └─────────────────────────────────┘
            ↓
        ┌─────────────────────────────────┐
        │ Fully Connected Layers          │
        ├─────────────────────────────────┤
        │ Dense(256) + BN + ReLU          │
        │ Dropout(0.5)                    │
        │ Dense(43) + Softmax             │
        └─────────────────────────────────┘
            ↓
        Output: 43 classes

Residual Block Structure:
        x → Conv → BN → ReLU → Conv → BN → (+) → ReLU
        |                                    ↑
        └────────────────────────────────────┘
              (shortcut/skip connection)

Key Features:
- Residual connections solve vanishing gradient problem
- Identity mapping allows learning residuals F(x) = H(x) - x
- 6 residual blocks across 3 stages
- Filter progression: 32 → 64 → 128
- Progressive dropout: 0.2 → 0.2 → 0.3 → 0.5
- Global Average Pooling reduces parameters
- ~800K parameters
- Expected accuracy: 93-96% (highest!)
- Stable training with deep networks

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

EPOCHS = 20  # ResNet can be trained for more epochs
IMG_WIDTH = 30
IMG_HEIGHT = 30
NUM_CATEGORIES = 43
TEST_SIZE = 0.4


def main():

    # Check command-line arguments
    if len(sys.argv) not in [2, 3]:
        sys.exit("Usage: python traffic_v5_resnet.py data_directory [model.h5]")

    # Get image arrays and labels for all image files
    print("=" * 70)
    print("Traffic Sign Recognition - ResNet Style (Version 5)")
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
    print("\nBuilding ResNet-style model...")
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


def residual_block(x, filters, kernel_size=3, stride=1, use_projection=False):
    """
    ResNet Residual Block
    
    Structure:
        x → Conv → BN → ReLU → Conv → BN → (+) → ReLU
        |                                    ↑
        └────────────────────────────────────┘
                    (shortcut/skip connection)
    
    Arguments:
        x: Input tensor
        filters: Number of convolution filters
        kernel_size: Size of the convolution kernel
        stride: Stride
        use_projection: Whether to use a projection shortcut (when dimensions do not match)
    """
    
    # Save the input (shortcut)
    shortcut = x
    
    # Main path
    # Conv -> BN -> ReLU
    x = tf.keras.layers.Conv2D(
        filters, kernel_size, strides=stride, padding='same'
    )(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    
    # Conv -> BN
    x = tf.keras.layers.Conv2D(filters, kernel_size, padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    
    # If dimensions do not match, use a 1x1 convolution to adjust the shortcut
    if use_projection:
        shortcut = tf.keras.layers.Conv2D(
            filters, (1, 1), strides=stride, padding='same'
        )(shortcut)
        shortcut = tf.keras.layers.BatchNormalization()(shortcut)
    
    # Residual connection: Add the input to the output
    x = tf.keras.layers.Add()([x, shortcut])
    x = tf.keras.layers.Activation('relu')(x)
    
    return x


def bottleneck_block(x, filters, stride=1, use_projection=False):
    """
    ResNet Bottleneck Block (for deeper networks)
    
    Structure:
        x → 1x1 Conv → BN → ReLU → 3x3 Conv → BN → ReLU → 1x1 Conv → BN → (+) → ReLU
        |                                                                    ↑
        └────────────────────────────────────────────────────────────────────┘
    
    Features:
        - Uses 1x1 convolutions for dimensionality reduction and restoration
        - Reduces the number of parameters
        - Suitable for deep networks
    """
    
    shortcut = x
    
    # 1x1 convolution for dimensionality reduction
    x = tf.keras.layers.Conv2D(filters, (1, 1), strides=stride, padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    
    # 3x3 convolution
    x = tf.keras.layers.Conv2D(filters, (3, 3), padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    
    # 1x1 convolution for dimensionality restoration
    x = tf.keras.layers.Conv2D(filters * 4, (1, 1), padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    
    # Adjust shortcut dimensions
    if use_projection:
        shortcut = tf.keras.layers.Conv2D(
            filters * 4, (1, 1), strides=stride, padding='same'
        )(shortcut)
        shortcut = tf.keras.layers.BatchNormalization()(shortcut)
    
    # Residual connection
    x = tf.keras.layers.Add()([x, shortcut])
    x = tf.keras.layers.Activation('relu')(x)
    
    return x


def get_model():
    """
    ResNet-Style Architecture (similar to ResNet-18/34)
    
    Features:
    - Uses Residual Blocks
    - Skip Connections to solve vanishing gradients
    - Can train deeper networks
    - Batch Normalization for standardization
    
    Architecture:
        Input → Conv → BN → ReLU → MaxPool
        → [Residual Block × 2] (32 filters)
        → [Residual Block × 2] (64 filters)
        → [Residual Block × 2] (128 filters)
        → Global Average Pooling → Dense → Output
    """
    
    inputs = tf.keras.Input(shape=(IMG_WIDTH, IMG_HEIGHT, 3))
    
    # ============ Initial Convolutional Layer ============
    x = tf.keras.layers.Conv2D(32, (3, 3), padding='same')(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    
    # ============ Stage 1: 32 filters ============
    # First residual block (does not change dimensions)
    x = residual_block(x, filters=32, use_projection=False)
    # Second residual block
    x = residual_block(x, filters=32, use_projection=False)
    x = tf.keras.layers.Dropout(0.2)(x)
    
    # ============ Stage 2: 64 filters ============
    # Down-sampling (stride=2) + increase number of channels
    x = residual_block(x, filters=64, stride=2, use_projection=True)
    x = residual_block(x, filters=64, use_projection=False)
    x = tf.keras.layers.Dropout(0.2)(x)
    
    # ============ Stage 3: 128 filters ============
    # Down-sampling (stride=2) + increase number of channels
    x = residual_block(x, filters=128, stride=2, use_projection=True)
    x = residual_block(x, filters=128, use_projection=False)
    x = tf.keras.layers.Dropout(0.3)(x)
    
    # ============ Global Average Pooling ============
    # ResNet feature: Use Global Average Pooling instead of Flatten
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    
    # ============ Fully Connected Layers ============
    x = tf.keras.layers.Dense(256, activation='relu')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    
    # ============ Output Layer ============
    outputs = tf.keras.layers.Dense(NUM_CATEGORIES, activation='softmax')(x)
    
    # Create model
    model = tf.keras.Model(inputs=inputs, outputs=outputs, name='ResNet_Style')
    
    # Compile the model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model


if __name__ == "__main__":
    main()