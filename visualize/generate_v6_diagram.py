"""
Generate V6 (Vision Transformer) architecture diagram
This script builds the model from code instead of loading from .h5
to avoid custom layer serialization issues.

Usage: python generate_v6_diagram.py
"""

import os
import sys
from tensorflow.keras.utils import plot_model

# Add traffic directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'traffic'))

# Import the model builder
from traffic_v6_vit import get_model

def main():
    print("=" * 70)
    print("Generating V6 (Vision Transformer) Architecture Diagram")
    print("=" * 70)
    
    # Create output directory
    output_dir = "architecture_diagrams"
    os.makedirs(output_dir, exist_ok=True)
    
    print("\nBuilding V6 model from code...")
    try:
        model = get_model()
        print("✓ Model built successfully")
    except Exception as e:
        print(f"✗ Error building model: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Generate diagram
    output_path = os.path.join(output_dir, "V6_ViT_Transformer.png")
    print(f"\nGenerating architecture diagram: {output_path}")
    
    try:
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
        print(f"✓ Diagram saved to {output_path}")
    except Exception as e:
        print(f"✗ Error generating diagram: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Print model summary
    print("\n" + "=" * 70)
    print("Model Summary:")
    print("=" * 70)
    model.summary()
    
    print("\n" + "=" * 70)
    print("Success!")
    print("=" * 70)
    print(f"\nDiagram saved to: {output_path}")
    print("\nNote: This diagram was generated from code, not from a saved .h5 file,")
    print("to avoid issues with custom layer serialization in Vision Transformer.")

if __name__ == "__main__":
    main()
