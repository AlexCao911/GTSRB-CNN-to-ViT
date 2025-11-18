"""
Training utilities for traffic sign recognition models

This module provides a unified training pipeline that eliminates code duplication
across different model architectures. It handles:
- Data loading and preprocessing
- Model training with configurable callbacks
- Model evaluation
- Model saving
- Progress reporting

Usage:
    from utils.trainer import ModelTrainer
    
    trainer = ModelTrainer(
        data_dir="gtsrb",
        model_name="ResNet-v5",
        epochs=20,
        test_size=0.4
    )
    
    model = get_model()  # Your model architecture
    trainer.train_and_evaluate(model, save_path="model_v5.h5")
"""

import numpy as np
import sys
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow import keras
from typing import Optional, Tuple, Dict, List, Callable

from .load_data import load_data


class ModelTrainer:
    """
    Unified training pipeline for traffic sign recognition models
    
    Attributes:
        data_dir: Path to the dataset directory
        model_name: Name of the model (for display purposes)
        epochs: Number of training epochs
        test_size: Fraction of data to use for testing (0.0 to 1.0)
        img_width: Width of input images
        img_height: Height of input images
        num_categories: Number of output classes
        batch_size: Batch size for training
        validation_split: Fraction of training data to use for validation
    """
    
    def __init__(
        self,
        data_dir: str,
        model_name: str = "CNN",
        epochs: int = 10,
        test_size: float = 0.4,
        img_width: int = 30,
        img_height: int = 30,
        num_categories: int = 43,
        batch_size: int = 32,
        validation_split: float = 0.2,
        verbose: bool = True
    ):
        self.data_dir = data_dir
        self.model_name = model_name
        self.epochs = epochs
        self.test_size = test_size
        self.img_width = img_width
        self.img_height = img_height
        self.num_categories = num_categories
        self.batch_size = batch_size
        self.validation_split = validation_split
        self.verbose = verbose
        
        # Data containers
        self.x_train = None
        self.x_test = None
        self.y_train = None
        self.y_test = None
        self.history = None
        
    def print_header(self):
        """Print a formatted header with model information"""
        if not self.verbose:
            return
            
        print("=" * 70)
        print(f"Traffic Sign Recognition - {self.model_name}")
        print("=" * 70)
        print()
        
    def load_and_prepare_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Load and preprocess the dataset
        
        Returns:
            Tuple of (x_train, x_test, y_train, y_test)
        """
        if self.verbose:
            print("Loading data...")
            
        # Load images and labels
        images, labels = load_data(
            self.data_dir,
            img_width=self.img_width,
            img_height=self.img_height,
            num_categories=self.num_categories
        )
        
        # Check if data was loaded successfully
        if len(images) == 0:
            print("\n❌ Error: No images loaded!")
            print(f"   Data directory: {self.data_dir}")
            print(f"   Please check:")
            print(f"   1. The directory exists")
            print(f"   2. The directory contains subdirectories 0-{self.num_categories-1}")
            print(f"   3. Each subdirectory contains image files")
            sys.exit(1)
            
        if self.verbose:
            print(f"✓ Loaded {len(images)} images from {len(set(labels))} categories")
            print("Preparing data...")
        
        # Normalize pixel values from 0-255 to 0-1
        images = np.array(images) / 255.0
        
        # Convert labels to categorical (one-hot encoding)
        labels = keras.utils.to_categorical(labels, num_classes=self.num_categories)
        
        # Split into training and testing sets
        x_train, x_test, y_train, y_test = train_test_split(
            images, labels, test_size=self.test_size, random_state=42
        )
        
        if self.verbose:
            print(f"✓ Training set: {len(x_train)} images")
            print(f"✓ Testing set: {len(x_test)} images")
            print()
        
        # Store for later use
        self.x_train = x_train
        self.x_test = x_test
        self.y_train = y_train
        self.y_test = y_test
        
        return x_train, x_test, y_train, y_test
    
    def get_callbacks(
        self,
        use_early_stopping: bool = False,
        use_lr_schedule: bool = False,
        use_model_checkpoint: bool = False,
        checkpoint_path: Optional[str] = None,
        custom_callbacks: Optional[List[keras.callbacks.Callback]] = None
    ) -> List[keras.callbacks.Callback]:
        """
        Get training callbacks based on configuration
        
        Args:
            use_early_stopping: Enable early stopping
            use_lr_schedule: Enable learning rate reduction on plateau
            use_model_checkpoint: Save best model during training
            checkpoint_path: Path to save checkpoints
            custom_callbacks: Additional custom callbacks
            
        Returns:
            List of Keras callbacks
        """
        callbacks = []
        
        if use_early_stopping:
            early_stop = keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=8,
                restore_best_weights=True,
                verbose=1 if self.verbose else 0
            )
            callbacks.append(early_stop)
        
        if use_lr_schedule:
            lr_schedule = keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss",
                factor=0.5,
                patience=3,
                min_lr=1e-6,
                verbose=1 if self.verbose else 0
            )
            callbacks.append(lr_schedule)
        
        if use_model_checkpoint and checkpoint_path:
            checkpoint = keras.callbacks.ModelCheckpoint(
                checkpoint_path,
                monitor="val_accuracy",
                save_best_only=True,
                verbose=1 if self.verbose else 0
            )
            callbacks.append(checkpoint)
        
        if custom_callbacks:
            callbacks.extend(custom_callbacks)
        
        return callbacks if callbacks else None
    
    def train_model(
        self,
        model: keras.Model,
        use_callbacks: bool = False,
        **callback_kwargs
    ) -> keras.callbacks.History:
        """
        Train the model
        
        Args:
            model: Compiled Keras model
            use_callbacks: Whether to use training callbacks
            **callback_kwargs: Arguments for get_callbacks()
            
        Returns:
            Training history object
        """
        if self.x_train is None:
            raise ValueError("Data not loaded. Call load_and_prepare_data() first.")
        
        if self.verbose:
            print("Training model...")
        
        # Get callbacks if requested
        callbacks = None
        if use_callbacks:
            callbacks = self.get_callbacks(**callback_kwargs)
        
        # Train the model
        history = model.fit(
            self.x_train,
            self.y_train,
            epochs=self.epochs,
            batch_size=self.batch_size,
            validation_split=self.validation_split,
            callbacks=callbacks,
            verbose=1 if self.verbose else 0
        )
        
        self.history = history
        return history
    
    def evaluate_model(self, model: keras.Model) -> Tuple[float, float]:
        """
        Evaluate the model on test data
        
        Args:
            model: Trained Keras model
            
        Returns:
            Tuple of (test_loss, test_accuracy)
        """
        if self.x_test is None:
            raise ValueError("Data not loaded. Call load_and_prepare_data() first.")
        
        if self.verbose:
            print("\nEvaluating model...")
        
        test_loss, test_acc = model.evaluate(
            self.x_test,
            self.y_test,
            verbose=2 if self.verbose else 0
        )
        
        if self.verbose:
            print(f"\n✓ Test Accuracy: {test_acc:.4f} ({test_acc*100:.2f}%)")
            print(f"✓ Test Loss: {test_loss:.4f}")
        
        return test_loss, test_acc
    
    def save_model(self, model: keras.Model, save_path: str):
        """
        Save the trained model
        
        Args:
            model: Trained Keras model
            save_path: Path to save the model
        """
        model.save(save_path)
        if self.verbose:
            print(f"\n✓ Model saved to {save_path}")
    
    def train_and_evaluate(
        self,
        model: keras.Model,
        save_path: Optional[str] = None,
        show_summary: bool = False,
        use_callbacks: bool = False,
        **callback_kwargs
    ) -> Dict[str, any]:
        """
        Complete training pipeline: load data, train, evaluate, and save
        
        Args:
            model: Compiled Keras model
            save_path: Path to save the trained model (optional)
            show_summary: Whether to print model summary
            use_callbacks: Whether to use training callbacks
            **callback_kwargs: Arguments for get_callbacks()
            
        Returns:
            Dictionary containing training results
        """
        # Print header
        self.print_header()
        
        # Load and prepare data
        self.load_and_prepare_data()
        
        # Show model summary if requested
        if show_summary and self.verbose:
            print("Building model...")
            model.summary()
            print("✓ Model created\n")
        elif self.verbose:
            print("Building model...")
            print("✓ Model created\n")
        
        # Train the model
        history = self.train_model(model, use_callbacks=use_callbacks, **callback_kwargs)
        
        # Evaluate the model
        test_loss, test_acc = self.evaluate_model(model)
        
        # Save the model if path provided
        if save_path:
            self.save_model(model, save_path)
        
        if self.verbose:
            print("\n✓ Training complete!")
        
        # Return results
        return {
            "history": history,
            "test_loss": test_loss,
            "test_accuracy": test_acc,
            "model": model
        }


def parse_training_args(default_epochs: int = 10) -> Tuple[str, Optional[str], int]:
    """
    Parse command-line arguments for training scripts
    
    Args:
        default_epochs: Default number of epochs if not specified
        
    Returns:
        Tuple of (data_dir, save_path, epochs)
        
    Usage:
        data_dir, save_path, epochs = parse_training_args(default_epochs=20)
    """
    if len(sys.argv) not in [2, 3]:
        print("Usage: python script.py data_directory [model.h5]")
        sys.exit(1)
    
    data_dir = sys.argv[1]
    save_path = sys.argv[2] if len(sys.argv) == 3 else None
    
    return data_dir, save_path, default_epochs


# Convenience function for simple training
def train_model_simple(
    model_builder: Callable[[], keras.Model],
    data_dir: str,
    save_path: Optional[str] = None,
    model_name: str = "CNN",
    epochs: int = 10,
    **trainer_kwargs
) -> Dict[str, any]:
    """
    Simplified training function for quick model training
    
    Args:
        model_builder: Function that returns a compiled Keras model
        data_dir: Path to dataset directory
        save_path: Path to save trained model (optional)
        model_name: Name of the model for display
        epochs: Number of training epochs
        **trainer_kwargs: Additional arguments for ModelTrainer
        
    Returns:
        Dictionary containing training results
        
    Example:
        def get_model():
            model = Sequential([...])
            model.compile(...)
            return model
        
        results = train_model_simple(
            get_model,
            data_dir="gtsrb",
            save_path="model.h5",
            model_name="MyModel",
            epochs=10
        )
    """
    trainer = ModelTrainer(
        data_dir=data_dir,
        model_name=model_name,
        epochs=epochs,
        **trainer_kwargs
    )
    
    model = model_builder()
    
    return trainer.train_and_evaluate(
        model,
        save_path=save_path,
        show_summary=True
    )
