"""
Traffic Sign Recognition - Version 6: Vision Transformer (ViT)
State-of-the-art transformer-based architecture for computer vision

Architecture Overview:
===========================================================================
        Input (30×30×3)
            ↓
        ┌─────────────────────────────────────────────────────────┐
        │ Patch Embedding                                         │
        ├─────────────────────────────────────────────────────────┤
        │ Split image into patches: 5×5 patches of 6×6 pixels     │
        │ Flatten each patch: 6×6×3 = 108 dimensions              │
        │ Linear projection: 108 → 128 (embedding_dim)            │
        │ Add positional encoding                                 │
        │ Output: (25 patches, 128 dim) + [CLS] token             │
        └─────────────────────────────────────────────────────────┘
            ↓
        ┌─────────────────────────────────────────────────────────┐
        │ Transformer Encoder × 4 layers                          │
        ├─────────────────────────────────────────────────────────┤
        │ Each layer contains:                                    │
        │   ┌─────────────────────────────────────┐               │
        │   │ Multi-Head Self-Attention (4 heads) │               │
        │   │   Q, K, V projections               │               │
        │   │   Attention(Q,K,V) = softmax(QK^T/√d)V │            │
        │   └─────────────────────────────────────┘               │
        │            ↓                                            │
        │   ┌─────────────────────────────────────┐               │
        │   │ Add & Norm (Residual + LayerNorm)   │               │
        │   └─────────────────────────────────────┘               │
        │            ↓                                            │
        │   ┌─────────────────────────────────────┐               │
        │   │ Feed-Forward Network (MLP)          │               │
        │   │   Dense(512) → GELU → Dense(128)    │               │
        │   └─────────────────────────────────────┘               │
        │            ↓                                            │
        │   ┌─────────────────────────────────────┐               │
        │   │ Add & Norm (Residual + LayerNorm)   │               │
        │   └─────────────────────────────────────┘               │
        └─────────────────────────────────────────────────────────┘
            ↓
        ┌─────────────────────────────────┐
        │ Classification Head             │
        ├─────────────────────────────────┤
        │ Extract [CLS] token             │
        │ LayerNorm                       │
        │ Dense(43) + Softmax             │
        └─────────────────────────────────┘
            ↓
        Output: 43 classes

Key Concepts:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. Patch Embedding:
   - Split 30×30 image into 5×5 = 25 patches
   - Each patch is 6×6 pixels
   - Flatten and project to embedding dimension

2. Self-Attention:
   - Each patch attends to all other patches
   - Learns global relationships
   - No inductive bias (unlike CNNs)

3. Positional Encoding:
   - Adds position information to patches
   - Learnable embeddings

4. [CLS] Token:
   - Special classification token
   - Aggregates information from all patches
   - Used for final classification

Features:
- Pure transformer architecture (no convolutions!)
- Global receptive field from layer 1
- Self-attention mechanism
- Positional encoding
- ~400K parameters
- Requires more data than CNNs
- State-of-the-art architecture

Comparison with CNNs:
- CNNs: Local receptive field → gradually expand
- ViT: Global receptive field from the start
- CNNs: Inductive bias (translation invariance)
- ViT: Learn everything from data

===========================================================================
"""

import cv2
import numpy as np
import os
import sys
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from sklearn.model_selection import train_test_split

# Add parent directory to path to import utils
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import load_data

EPOCHS = 30  # ViT needs more epochs
IMG_WIDTH = 30
IMG_HEIGHT = 30
NUM_CATEGORIES = 43
TEST_SIZE = 0.4

# ViT Hyperparameters (optimized for small images and dataset)
PATCH_SIZE = 5  # Smaller patches for 30×30 images: 6×6 patches
NUM_PATCHES = (IMG_WIDTH // PATCH_SIZE) * (IMG_HEIGHT // PATCH_SIZE)  # 6×6 = 36
EMBEDDING_DIM = 64  # Smaller embedding for small dataset
NUM_HEADS = 4  # Number of attention heads
TRANSFORMER_LAYERS = 3  # Fewer layers for small dataset
MLP_DIM = 256  # Smaller MLP for small dataset


def main():

    # Check command-line arguments
    if len(sys.argv) not in [2, 3]:
        sys.exit("Usage: python traffic_v6_vit.py data_directory [model.h5]")

    # Get image arrays and labels for all image files
    print("=" * 70)
    print("Traffic Sign Recognition - Vision Transformer (Version 6)")
    print("=" * 70)
    print("\nLoading data...")
    images, labels = load_data(sys.argv[1])
    print(f"✓ Loaded {len(images)} images from {len(set(labels))} categories")

    # Check if data was loaded
    if len(images) == 0:
        print("\n❌ Error: No images loaded!")
        print(f"   Data directory: {sys.argv[1]}")
        print(f"   Please check:")
        print(f"   1. The directory exists")
        print(f"   2. The directory contains subdirectories 0-42")
        print(f"   3. Each subdirectory contains image files")
        print(f"\n   Usage: python traffic_v6_vit.py <data_directory> [model.h5]")
        print(f"   Example: python traffic_v6_vit.py gtsrb model_v6.h5")
        sys.exit(1)

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
    print("\nBuilding Vision Transformer model...")
    model = get_model()
    model.summary()
    print("✓ Model created\n")

    # Fit model on training data with callbacks
    print("Training model...")
    
    # Learning rate schedule
    lr_schedule = keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=3,
        min_lr=1e-6,
        verbose=1,
    )
    
    # Early stopping
    early_stop = keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=8, restore_best_weights=True, verbose=1
    )
    
    history = model.fit(
        x_train,
        y_train,
        epochs=EPOCHS,
        validation_split=0.2,
        batch_size=32,  # Smaller batch size for ViT
        callbacks=[lr_schedule, early_stop],
        verbose=1,
    )

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


class PatchEmbedding(layers.Layer):
    """
    Convert image into patches and embed them
    
    Process:
    1. Extract patches from image
    2. Flatten each patch
    3. Linear projection to embedding dimension
    4. Add positional encoding
    """

    def __init__(self, patch_size, embedding_dim, **kwargs):
        super().__init__(**kwargs)
        self.patch_size = patch_size
        self.embedding_dim = embedding_dim

    def build(self, input_shape):
        # Linear projection of flattened patches with better initialization
        self.projection = layers.Dense(
            self.embedding_dim,
            kernel_initializer=keras.initializers.TruncatedNormal(stddev=0.02),
        )

        # Positional encoding (learnable)
        self.position_embedding = layers.Embedding(
            input_dim=NUM_PATCHES + 1,
            output_dim=self.embedding_dim,
            embeddings_initializer=keras.initializers.TruncatedNormal(stddev=0.02),
        )

        # [CLS] token (learnable) with better initialization
        self.cls_token = self.add_weight(
            shape=(1, 1, self.embedding_dim),
            initializer=keras.initializers.TruncatedNormal(stddev=0.02),
            trainable=True,
            name="cls_token",
        )

        super().build(input_shape)

    def call(self, images):
        batch_size = tf.shape(images)[0]

        # Extract patches
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding="VALID",
        )

        # Reshape: (batch, num_patches, patch_size*patch_size*channels)
        patch_dims = patches.shape[-1]
        patches = tf.reshape(patches, [batch_size, -1, patch_dims])

        # Linear projection
        encoded_patches = self.projection(patches)

        # Add [CLS] token
        cls_tokens = tf.broadcast_to(self.cls_token, [batch_size, 1, self.embedding_dim])
        encoded_patches = tf.concat([cls_tokens, encoded_patches], axis=1)

        # Add positional encoding
        positions = tf.range(start=0, limit=NUM_PATCHES + 1, delta=1)
        position_embeddings = self.position_embedding(positions)
        encoded_patches = encoded_patches + position_embeddings

        return encoded_patches


class MultiHeadSelfAttention(layers.Layer):
    """
    Multi-Head Self-Attention mechanism
    
    Attention(Q, K, V) = softmax(QK^T / √d_k) V
    
    Where:
    - Q: Query
    - K: Key
    - V: Value
    - d_k: Dimension of key
    """

    def __init__(self, embedding_dim, num_heads, **kwargs):
        super().__init__(**kwargs)
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.head_dim = embedding_dim // num_heads

        assert (
            self.head_dim * num_heads == embedding_dim
        ), "Embedding dim must be divisible by num_heads"

        # Linear projections for Q, K, V
        self.query_dense = layers.Dense(embedding_dim)
        self.key_dense = layers.Dense(embedding_dim)
        self.value_dense = layers.Dense(embedding_dim)

        # Output projection
        self.combine_heads = layers.Dense(embedding_dim)

    def attention(self, query, key, value):
        # Calculate attention scores
        score = tf.matmul(query, key, transpose_b=True)
        dim_key = tf.cast(tf.shape(key)[-1], tf.float32)
        scaled_score = score / tf.math.sqrt(dim_key)

        # Apply softmax
        weights = tf.nn.softmax(scaled_score, axis=-1)

        # Apply attention weights to values
        output = tf.matmul(weights, value)
        return output, weights

    def separate_heads(self, x, batch_size):
        # Reshape to (batch, num_patches, num_heads, head_dim)
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.head_dim))
        # Transpose to (batch, num_heads, num_patches, head_dim)
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, inputs):
        batch_size = tf.shape(inputs)[0]

        # Linear projections
        query = self.query_dense(inputs)
        key = self.key_dense(inputs)
        value = self.value_dense(inputs)

        # Separate heads
        query = self.separate_heads(query, batch_size)
        key = self.separate_heads(key, batch_size)
        value = self.separate_heads(value, batch_size)

        # Apply attention
        attention_output, _ = self.attention(query, key, value)

        # Concatenate heads
        attention_output = tf.transpose(attention_output, perm=[0, 2, 1, 3])
        concat_attention = tf.reshape(
            attention_output, (batch_size, -1, self.embedding_dim)
        )

        # Final linear projection
        output = self.combine_heads(concat_attention)
        return output


class TransformerBlock(layers.Layer):
    """
    Transformer Encoder Block
    
    Structure:
    1. Multi-Head Self-Attention
    2. Add & Norm (Residual + LayerNorm)
    3. Feed-Forward Network (MLP)
    4. Add & Norm (Residual + LayerNorm)
    """

    def __init__(self, embedding_dim, num_heads, mlp_dim, dropout_rate=0.2, **kwargs):
        super().__init__(**kwargs)
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.mlp_dim = mlp_dim

        # Multi-Head Self-Attention
        self.attention = MultiHeadSelfAttention(embedding_dim, num_heads)
        self.attention_dropout = layers.Dropout(dropout_rate)
        self.attention_norm = layers.LayerNormalization(epsilon=1e-6)

        # Feed-Forward Network (MLP) with better initialization
        self.mlp = keras.Sequential(
            [
                layers.Dense(
                    mlp_dim,
                    activation=tf.nn.gelu,
                    kernel_initializer=keras.initializers.TruncatedNormal(stddev=0.02),
                ),
                layers.Dropout(dropout_rate),
                layers.Dense(
                    embedding_dim,
                    kernel_initializer=keras.initializers.TruncatedNormal(stddev=0.02),
                ),
                layers.Dropout(dropout_rate),
            ]
        )
        self.mlp_norm = layers.LayerNormalization(epsilon=1e-6)

    def call(self, inputs, training=False):
        # Multi-Head Self-Attention with residual connection
        attention_output = self.attention(inputs)
        attention_output = self.attention_dropout(attention_output, training=training)
        x1 = self.attention_norm(inputs + attention_output)

        # Feed-Forward Network with residual connection
        mlp_output = self.mlp(x1, training=training)
        x2 = self.mlp_norm(x1 + mlp_output)

        return x2


def get_model():
    """
    Vision Transformer (ViT) Architecture
    
    Components:
    1. Patch Embedding
    2. Transformer Encoder × N
    3. Classification Head
    """

    inputs = layers.Input(shape=(IMG_WIDTH, IMG_HEIGHT, 3))

    # ============ Patch Embedding ============
    patches = PatchEmbedding(PATCH_SIZE, EMBEDDING_DIM)(inputs)

    # ============ Transformer Encoder ============
    x = patches
    for _ in range(TRANSFORMER_LAYERS):
        x = TransformerBlock(EMBEDDING_DIM, NUM_HEADS, MLP_DIM, dropout_rate=0.2)(x)

    # ============ Classification Head ============
    # Extract [CLS] token (first token)
    x = layers.LayerNormalization(epsilon=1e-6)(x)
    cls_token = x[:, 0]  # Shape: (batch_size, embedding_dim)

    # Classification with better initialization
    x = layers.Dropout(0.3)(cls_token)
    outputs = layers.Dense(
        NUM_CATEGORIES,
        activation="softmax",
        kernel_initializer=keras.initializers.TruncatedNormal(stddev=0.02),
    )(x)

    # Create model
    model = keras.Model(inputs=inputs, outputs=outputs, name="ViT")

    # Compile with lower learning rate and warmup
    # Lower learning rate is crucial for ViT
    model.compile(
        optimizer=keras.optimizers.Adam(
            learning_rate=0.0001,  # Much lower learning rate!
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-8,
        ),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model


if __name__ == "__main__":
    main()
