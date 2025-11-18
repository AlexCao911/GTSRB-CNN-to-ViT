"""
Traffic Sign Recognition - Version 5: ResNet Style (Refactored)
Revolutionary residual learning architecture

This is a refactored version using the unified training pipeline.

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
        Global Average Pooling  (128)
            ↓
        Dense(256) + BN + Dropout(0.5)
            ↓
        Dense(43) + Softmax

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
- ~744K parameters
- Expected accuracy: 95-96%
- Stable training with deep networks

===========================================================================
"""

import os
import sys
import tensorflow as tf

# Add parent directory to path to import utils
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import ModelTrainer, parse_training_args

# Model configuration
EPOCHS = 20  # ResNet can be trained for more epochs
IMG_WIDTH = 30
IMG_HEIGHT = 30
NUM_CATEGORIES = 43
TEST_SIZE = 0.4


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
        stride: Stride for downsampling
        use_projection: Whether to use a projection shortcut (when dimensions change)
        
    Returns:
        Output tensor after residual connection
    """
    
    # Save the input (shortcut)
    shortcut = x
    
    # Main path
    # Conv → BN → ReLU
    x = tf.keras.layers.Conv2D(
        filters, kernel_size, strides=stride, padding='same'
    )(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    
    # Conv → BN
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


def get_model():
    """
    ResNet-Style Architecture (similar to ResNet-18/34)
    
    Features:
    - Uses Residual Blocks with skip connections
    - Solves vanishing gradient problem
    - Can train deeper networks
    - Batch Normalization for standardization
    - Global Average Pooling
    
    Architecture:
        Input → Conv → BN → ReLU
        → [Residual Block × 2] (32 filters)
        → [Residual Block × 2] (64 filters, stride=2 for first)
        → [Residual Block × 2] (128 filters, stride=2 for first)
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


def main():
    # Parse command-line arguments
    data_dir, save_path, _ = parse_training_args(default_epochs=EPOCHS)
    
    # Create trainer
    trainer = ModelTrainer(
        data_dir=data_dir,
        model_name="ResNet-Style (Residual Learning) - Version 5",
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
        show_summary=True  # Show architecture for ResNet
    )


if __name__ == "__main__":
    main()
