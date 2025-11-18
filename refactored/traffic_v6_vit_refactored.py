"""
Traffic Sign Recognition - Version 6: Vision Transformer (Refactored)
State-of-the-art transformer-based architecture for computer vision

This is a refactored version using the unified training pipeline.
Demonstrates advanced usage with callbacks and custom training configuration.

Architecture Overview:
===========================================================================
        Input (30×30×3)
            ↓
        Patch Embedding (36 patches of 5×5)
            ↓
        Transformer Encoder × 3 layers
            ↓
        Classification Head
            ↓
        Output: 43 classes

Key Concepts:
- Patch Embedding: Split image into patches
- Self-Attention: Each patch attends to all other patches
- Positional Encoding: Adds position information
- [CLS] Token: Special classification token

Features:
- Pure transformer architecture (no convolutions!)
- Global receptive field from layer 1
- ~160K parameters
- Requires more data than CNNs
- Expected accuracy: 92-93%

===========================================================================
"""

import os
import sys
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Add parent directory to path to import utils
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import ModelTrainer, parse_training_args

# Model configuration
EPOCHS = 30  # ViT needs more epochs
IMG_WIDTH = 30
IMG_HEIGHT = 30
NUM_CATEGORIES = 43
TEST_SIZE = 0.4

# ViT Hyperparameters
PATCH_SIZE = 5
NUM_PATCHES = (IMG_WIDTH // PATCH_SIZE) * (IMG_HEIGHT // PATCH_SIZE)
EMBEDDING_DIM = 64
NUM_HEADS = 4
TRANSFORMER_LAYERS = 3
MLP_DIM = 256


class PatchEmbedding(layers.Layer):
    """Convert image into patches and embed them"""

    def __init__(self, patch_size, embedding_dim, **kwargs):
        super().__init__(**kwargs)
        self.patch_size = patch_size
        self.embedding_dim = embedding_dim

    def build(self, input_shape):
        self.projection = layers.Dense(
            self.embedding_dim,
            kernel_initializer=keras.initializers.TruncatedNormal(stddev=0.02),
        )
        self.position_embedding = layers.Embedding(
            input_dim=NUM_PATCHES + 1,
            output_dim=self.embedding_dim,
            embeddings_initializer=keras.initializers.TruncatedNormal(stddev=0.02),
        )
        self.cls_token = self.add_weight(
            shape=(1, 1, self.embedding_dim),
            initializer=keras.initializers.TruncatedNormal(stddev=0.02),
            trainable=True,
            name="cls_token",
        )
        super().build(input_shape)

    def call(self, images):
        batch_size = tf.shape(images)[0]
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding="VALID",
        )
        patch_dims = patches.shape[-1]
        patches = tf.reshape(patches, [batch_size, -1, patch_dims])
        encoded_patches = self.projection(patches)
        cls_tokens = tf.broadcast_to(self.cls_token, [batch_size, 1, self.embedding_dim])
        encoded_patches = tf.concat([cls_tokens, encoded_patches], axis=1)
        positions = tf.range(start=0, limit=NUM_PATCHES + 1, delta=1)
        position_embeddings = self.position_embedding(positions)
        encoded_patches = encoded_patches + position_embeddings
        return encoded_patches


class MultiHeadSelfAttention(layers.Layer):
    """Multi-Head Self-Attention mechanism"""

    def __init__(self, embedding_dim, num_heads, **kwargs):
        super().__init__(**kwargs)
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.head_dim = embedding_dim // num_heads
        assert self.head_dim * num_heads == embedding_dim
        
        self.query_dense = layers.Dense(embedding_dim)
        self.key_dense = layers.Dense(embedding_dim)
        self.value_dense = layers.Dense(embedding_dim)
        self.combine_heads = layers.Dense(embedding_dim)

    def attention(self, query, key, value):
        score = tf.matmul(query, key, transpose_b=True)
        dim_key = tf.cast(tf.shape(key)[-1], tf.float32)
        scaled_score = score / tf.math.sqrt(dim_key)
        weights = tf.nn.softmax(scaled_score, axis=-1)
        output = tf.matmul(weights, value)
        return output, weights

    def separate_heads(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.head_dim))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, inputs):
        batch_size = tf.shape(inputs)[0]
        query = self.query_dense(inputs)
        key = self.key_dense(inputs)
        value = self.value_dense(inputs)
        query = self.separate_heads(query, batch_size)
        key = self.separate_heads(key, batch_size)
        value = self.separate_heads(value, batch_size)
        attention_output, _ = self.attention(query, key, value)
        attention_output = tf.transpose(attention_output, perm=[0, 2, 1, 3])
        concat_attention = tf.reshape(attention_output, (batch_size, -1, self.embedding_dim))
        output = self.combine_heads(concat_attention)
        return output


class TransformerBlock(layers.Layer):
    """Transformer Encoder Block"""

    def __init__(self, embedding_dim, num_heads, mlp_dim, dropout_rate=0.2, **kwargs):
        super().__init__(**kwargs)
        self.attention = MultiHeadSelfAttention(embedding_dim, num_heads)
        self.attention_dropout = layers.Dropout(dropout_rate)
        self.attention_norm = layers.LayerNormalization(epsilon=1e-6)
        self.mlp = keras.Sequential([
            layers.Dense(mlp_dim, activation=tf.nn.gelu,
                        kernel_initializer=keras.initializers.TruncatedNormal(stddev=0.02)),
            layers.Dropout(dropout_rate),
            layers.Dense(embedding_dim,
                        kernel_initializer=keras.initializers.TruncatedNormal(stddev=0.02)),
            layers.Dropout(dropout_rate),
        ])
        self.mlp_norm = layers.LayerNormalization(epsilon=1e-6)

    def call(self, inputs, training=False):
        attention_output = self.attention(inputs)
        attention_output = self.attention_dropout(attention_output, training=training)
        x1 = self.attention_norm(inputs + attention_output)
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
    
    # Patch Embedding
    patches = PatchEmbedding(PATCH_SIZE, EMBEDDING_DIM)(inputs)
    
    # Transformer Encoder
    x = patches
    for _ in range(TRANSFORMER_LAYERS):
        x = TransformerBlock(EMBEDDING_DIM, NUM_HEADS, MLP_DIM, dropout_rate=0.2)(x)
    
    # Classification Head
    x = layers.LayerNormalization(epsilon=1e-6)(x)
    cls_token = x[:, 0]
    x = layers.Dropout(0.3)(cls_token)
    outputs = layers.Dense(
        NUM_CATEGORIES,
        activation="softmax",
        kernel_initializer=keras.initializers.TruncatedNormal(stddev=0.02),
    )(x)
    
    # Create model
    model = keras.Model(inputs=inputs, outputs=outputs, name="ViT")
    
    # Compile with lower learning rate (crucial for ViT)
    model.compile(
        optimizer=keras.optimizers.Adam(
            learning_rate=0.0001,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-8,
        ),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    
    return model


def main():
    # Parse command-line arguments
    data_dir, save_path, _ = parse_training_args(default_epochs=EPOCHS)
    
    # Create trainer with custom configuration
    trainer = ModelTrainer(
        data_dir=data_dir,
        model_name="Vision Transformer (ViT) - Version 6",
        epochs=EPOCHS,
        test_size=TEST_SIZE,
        img_width=IMG_WIDTH,
        img_height=IMG_HEIGHT,
        num_categories=NUM_CATEGORIES,
        batch_size=32  # Smaller batch size for ViT
    )
    
    # Build model
    model = get_model()
    
    # Train and evaluate with callbacks
    # ViT benefits from early stopping and learning rate scheduling
    results = trainer.train_and_evaluate(
        model,
        save_path=save_path,
        show_summary=True,  # Show architecture for ViT
        use_callbacks=True,  # Enable callbacks
        use_early_stopping=True,  # Stop if no improvement
        use_lr_schedule=True  # Reduce LR on plateau
    )


if __name__ == "__main__":
    main()
