"""
Traffic Sign Recognition - Version 2: Improved CNN (Refactored)
AlexNet-inspired architecture with dual convolutional blocks

This is a refactored version using the unified training pipeline.
All boilerplate code has been moved to utils.trainer module.

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
- Expected accuracy: 99%+ ⭐
- Best balance of performance and efficiency

===========================================================================
"""

import os
import sys
import tensorflow as tf

# Add parent directory to path to import utils
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import ModelTrainer, parse_training_args

# Model configuration
EPOCHS = 10
IMG_WIDTH = 30
IMG_HEIGHT = 30
NUM_CATEGORIES = 43
TEST_SIZE = 0.4


def get_model():
    """
    Returns a compiled convolutional neural network model.
    
    Architecture:
    - Convolutional Block 1: Conv(32)×2 + MaxPool + Dropout
    - Convolutional Block 2: Conv(64)×2 + MaxPool + Dropout
    - Flatten
    - Dense(512) + Dropout
    - Dense(43) + Softmax
    """
    
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
     
    # Compile the CNN model
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
        model_name="Advanced CNN (AlexNet-inspired) - Version 2 ⭐",
        epochs=EPOCHS,
        test_size=TEST_SIZE,
        img_width=IMG_WIDTH,
        img_height=IMG_HEIGHT,
        num_categories=NUM_CATEGORIES
    )
    
    # Build model
    model = get_model()
    
    # Train and evaluate (all in one call!)
    results = trainer.train_and_evaluate(
        model,
        save_path=save_path,
        show_summary=False  # Set to True to see model architecture
    )


if __name__ == "__main__":
    main()
