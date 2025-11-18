"""
Traffic Sign Recognition - Version 3: VGG Style (Refactored)
VGG-inspired deep architecture with uniform 3x3 kernels

This is a refactored version using the unified training pipeline.

Architecture Overview:
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
        │ Dense(256) + ReLU               │
        │ BatchNorm + Dropout(0.5)        │
        │ Dense(43) + Softmax             │
        └─────────────────────────────────┘

Features:
- 3 convolutional blocks with 2 conv layers each
- Uniform 3x3 kernels (VGG's key innovation)
- Filter progression: 32 → 64 → 128
- padding='same' maintains spatial dimensions
- Batch Normalization for stable training
- Multiple dense layers
- ~1.0M parameters
- Expected accuracy: 99%+

===========================================================================
"""

import os
import sys
import tensorflow as tf

# Add parent directory to path to import utils
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import ModelTrainer, parse_training_args

# Model configuration
EPOCHS = 15  # VGG requires more training epochs
IMG_WIDTH = 30
IMG_HEIGHT = 30
NUM_CATEGORIES = 43
TEST_SIZE = 0.4


def get_model():
    """
    VGG-Style Architecture
    
    Features:
    - 3 convolutional blocks, each with 2 convolutional layers
    - Filters: 32 → 64 → 128
    - padding='same' to maintain feature map size
    - Uses Batch Normalization to speed up training
    - Deeper network structure
    
    Architecture:
    - Block 1: Conv(32)×2 + BN + MaxPool + Dropout
    - Block 2: Conv(64)×2 + BN + MaxPool + Dropout
    - Block 3: Conv(128)×2 + BN + MaxPool + Dropout
    - Flatten
    - Dense(512) + BN + Dropout
    - Dense(256) + BN + Dropout
    - Dense(43) + Softmax
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


def main():
    # Parse command-line arguments
    data_dir, save_path, _ = parse_training_args(default_epochs=EPOCHS)
    
    # Create trainer
    trainer = ModelTrainer(
        data_dir=data_dir,
        model_name="VGG-Style Deep Network - Version 3",
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
        show_summary=True  # Show architecture for VGG
    )


if __name__ == "__main__":
    main()
