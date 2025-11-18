"""
Traffic Sign Recognition - Version 1: Basic CNN (Refactored)
LeNet-inspired simple architecture

This is a refactored version using the unified training pipeline.
All boilerplate code has been moved to utils.trainer module.

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
- ~809K parameters
- Good for quick prototyping
- Expected accuracy: 96-97%

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
    - Conv2D(32, 3x3) + ReLU
    - MaxPooling2D(2x2)
    - Flatten
    - Dense(128) + ReLU
    - Dropout(0.5)
    - Dense(43) + Softmax
    """
    
    model = tf.keras.models.Sequential([
        # Convolutional layer 
        tf.keras.layers.Conv2D(
            32, (3, 3), activation="relu", input_shape=(IMG_WIDTH, IMG_HEIGHT, 3)
        ),
        
        # Max-pool layer, using 2x2 pool size 
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        
        # Flatten units
        tf.keras.layers.Flatten(),
        
        # Add a hidden layer with dropout
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dropout(0.5),
        
        # Add an output layer 
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
        model_name="Basic CNN (LeNet-inspired) - Version 1",
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
    
    # Access results if needed
    # print(f"Final accuracy: {results['test_accuracy']:.4f}")
    # print(f"Training history: {results['history'].history}")


if __name__ == "__main__":
    main()
