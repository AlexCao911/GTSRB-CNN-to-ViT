"""
Simple script to generate architecture diagrams from trained models
Usage: python generate_diagrams.py
"""

import os
import sys
import tensorflow as tf
from tensorflow.keras.utils import plot_model

# Add traffic directory to path for custom layers
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'traffic'))

def get_custom_objects():
    """
    Import custom layers from V6 (ViT) model
    """
    try:
        from traffic_v6_vit import PatchEmbedding, MultiHeadSelfAttention, TransformerBlock
        return {
            'PatchEmbedding': PatchEmbedding,
            'MultiHeadSelfAttention': MultiHeadSelfAttention,
            'TransformerBlock': TransformerBlock
        }
    except ImportError as e:
        print(f"⚠ Warning: Could not import custom layers: {e}")
        return {}

def generate_diagram_from_h5(model_path, output_path):
    """Generate Archeture from .5 files"""
    try:
        print(f"Loading model: {model_path}")
        
        # Load model with custom objects for V6
        custom_objects = get_custom_objects()
        if custom_objects:
            model = tf.keras.models.load_model(model_path, custom_objects=custom_objects)
        else:
            model = tf.keras.models.load_model(model_path)
        
        print(f"Generating diagram: {output_path}")
        plot_model(
            model,
            to_file=output_path,
            show_shapes=True,
            show_layer_names=True,
            rankdir='TB',
            expand_nested=True,
            dpi=200,
            show_layer_activations=True
        )
        print(f"✓ Saved: {output_path}\n")
        return True
    except Exception as e:
        print(f"✗ Error: {e}\n")
        import traceback
        traceback.print_exc()
        return False

def main():
    
    output_dir = "architecture_diagrams"
    os.makedirs(output_dir, exist_ok=True)
    
    models = [
        ("model_v1.h5", "V1_LeNet_Basic.png"),
        ("model_v2.h5", "V2_AlexNet_Improved.png"),
        ("model_v3.h5", "V3_VGG_Deep.png"),
        ("model_v4.h5", "V4_GoogLeNet_Inception.png"),
        ("model_v5.h5", "V5_ResNet_Residual.png"),
    ]
    
    print("=" * 70)
    print("Generating Architecture Diagrams from Trained Models")
    print("=" * 70)
    print()
    
    success_count = 0
    for model_file, output_file in models:
        if os.path.exists(model_file):
            output_path = os.path.join(output_dir, output_file)
            if generate_diagram_from_h5(model_file, output_path):
                success_count += 1
        else:
            print(f"⚠ Model not found: {model_file}")
            print(f"  Train it first with: python traffic/traffic_v{model_file[7]}_*.py gtsrb {model_file}\n")
    
    print("=" * 70)
    print(f"Complete! Generated {success_count}/{len(models)} diagrams")
    print(f"Saved to: {output_dir}/")
    print("=" * 70)
    
    # Special note for V6
    print("\n" + "=" * 70)
    print("Note: V6 (Vision Transformer) Diagram")
    print("=" * 70)
    print("V6 uses custom layers that have serialization issues with .h5 files.")
    print("To generate V6 diagram, use:")
    print("  python generate_v6_diagram.py")
    print("=" * 70)

if __name__ == "__main__":
    main()
