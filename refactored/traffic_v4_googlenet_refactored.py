"""
Traffic Sign Recognition - Version 4: GoogLeNet (Inception) Style (Refactored)
Inception-inspired architecture with multi-scale feature extraction

This is a refactored version using the unified training pipeline.

Architecture Overview:
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
        Global Average Pooling  (144)
            ↓
        Dense(512) + BN + Dropout(0.5)
            ↓
        Dense(43) + Softmax

Features:
- Inception modules for multi-scale feature extraction
- 1x1 convolutions for dimensionality reduction
- Parallel processing at different scales
- Global Average Pooling (reduces parameters)
- ~148K parameters (most efficient!)
- Expected accuracy: 95-96%

===========================================================================
"""

import os
import sys
import tensorflow as tf

# Add parent directory to path to import utils
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import ModelTrainer, parse_training_args

# Model configuration
EPOCHS = 15
IMG_WIDTH = 30
IMG_HEIGHT = 30
NUM_CATEGORIES = 43
TEST_SIZE = 0.4


def inception_module(x, filters_1x1, filters_3x3_reduce, filters_3x3, 
                     filters_5x5_reduce, filters_5x5, filters_pool_proj):
    """
    Inception Module
    
    Performs 4 operations in parallel:
    1. 1x1 convolution
    2. 1x1 convolution → 3x3 convolution
    3. 1x1 convolution → 5x5 convolution
    4. 3x3 MaxPooling → 1x1 convolution
    
    Then concatenates the results along the channel dimension
    
    Args:
        x: Input tensor
        filters_1x1: Number of 1x1 filters in branch 1
        filters_3x3_reduce: Number of 1x1 filters before 3x3 in branch 2
        filters_3x3: Number of 3x3 filters in branch 2
        filters_5x5_reduce: Number of 1x1 filters before 5x5 in branch 3
        filters_5x5: Number of 5x5 filters in branch 3
        filters_pool_proj: Number of 1x1 filters after pooling in branch 4
        
    Returns:
        Concatenated output tensor
    """
    
    # Branch 1: 1x1 convolution
    branch1 = tf.keras.layers.Conv2D(filters_1x1, (1, 1), padding='same', activation='relu')(x)
    
    # Branch 2: 1x1 convolution → 3x3 convolution
    branch2 = tf.keras.layers.Conv2D(filters_3x3_reduce, (1, 1), padding='same', activation='relu')(x)
    branch2 = tf.keras.layers.Conv2D(filters_3x3, (3, 3), padding='same', activation='relu')(branch2)
    
    # Branch 3: 1x1 convolution → 5x5 convolution
    branch3 = tf.keras.layers.Conv2D(filters_5x5_reduce, (1, 1), padding='same', activation='relu')(x)
    branch3 = tf.keras.layers.Conv2D(filters_5x5, (5, 5), padding='same', activation='relu')(branch3)
    
    # Branch 4: 3x3 MaxPooling → 1x1 convolution
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
    - Global Average Pooling instead of Flatten
    - High parameter efficiency
    
    Architecture:
    - Initial Conv(32) + BN + MaxPool
    - Inception Module 1 (outputs 72 filters)
    - BN + Dropout + MaxPool
    - Inception Module 2 (outputs 144 filters)
    - BN + Dropout
    - Global Average Pooling
    - Dense(512) + BN + Dropout
    - Dense(43) + Softmax
    """
    
    # Use Functional API (Inception module requires branching structure)
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


def main():
    # Parse command-line arguments
    data_dir, save_path, _ = parse_training_args(default_epochs=EPOCHS)
    
    # Create trainer
    trainer = ModelTrainer(
        data_dir=data_dir,
        model_name="GoogLeNet/Inception-Style - Version 4",
        epochs=EPOCHS,
        test_size=TEST_SIZE,
        img_width=IMG_WIDTH,
        img_height=IMG_HEIGHT,
        num_categories=NUM_CATEGORIES
    )
    
    # Build model
    model = get_model()
    
    # Train and evaluate
    results = trainer.train_and_evaluate(
        model,
        save_path=save_path,
        show_summary=True  # Show architecture for Inception
    )


if __name__ == "__main__":
    main()
