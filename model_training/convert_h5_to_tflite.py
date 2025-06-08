#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
convert_h5_to_tflite.py - Convert Keras H5 model to TensorFlow Lite format
"""

import os
import sys
import argparse
import tensorflow as tf
import numpy as np
from tensorflow.keras.models import load_model
import datetime

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Convert Keras H5 model to TensorFlow Lite format')
    parser.add_argument('--input-model', type=str, required=True,
                        help='Path to the input H5 model file')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Output directory for the TFLite model (defaults to same directory as input model)')
    parser.add_argument('--output-name', type=str, default=None,
                        help='Name of the output TFLite model file (defaults to input model name with .tflite extension)')
    parser.add_argument('--quantize', action='store_true',
                        help='Apply post-training quantization to reduce model size')
    parser.add_argument('--representative-dataset', type=str, default=None,
                        help='Path to representative dataset for quantization (if using full integer quantization)')
    
    return parser.parse_args()

def get_output_path(args):
    """Get the output path for the TFLite model"""
    # Get the output directory
    if args.output_dir:
        output_dir = args.output_dir
    else:
        output_dir = os.path.dirname(args.input_model)
    
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Get the output file name
    if args.output_name:
        output_name = args.output_name
        if not output_name.endswith('.tflite'):
            output_name += '.tflite'
    else:
        input_basename = os.path.basename(args.input_model)
        output_name = os.path.splitext(input_basename)[0] + '.tflite'
    
    return os.path.join(output_dir, output_name)

def load_keras_model(model_path):
    """Load a Keras model from H5 file"""
    try:
        model = load_model(model_path)
        print(f"Loaded model from {model_path}")
        print(f"Model input shape: {model.input_shape}")
        print(f"Model output shape: {model.output_shape}")
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)

def create_representative_dataset(dataset_path):
    """Create a representative dataset for quantization"""
    if not os.path.exists(dataset_path):
        print(f"Error: Representative dataset path not found: {dataset_path}")
        sys.exit(1)
    
    # Simple implementation for image data
    # In a real scenario, this should be adapted to your specific data format
    image_files = [os.path.join(dataset_path, f) for f in os.listdir(dataset_path) 
                   if f.endswith(('.jpg', '.jpeg', '.png'))]
    
    def representative_data_gen():
        for image_file in image_files[:100]:  # Limit to 100 images
            img = tf.keras.preprocessing.image.load_img(image_file, target_size=(224, 224))
            img_array = tf.keras.preprocessing.image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)
            img_array = img_array.astype(np.float32) / 255.0
            yield [img_array]
    
    return representative_data_gen

def convert_to_tflite(model, output_path, quantize=False, representative_dataset=None):
    """Convert Keras model to TFLite format"""
    # Create TFLite converter
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    
    # Set conversion options
    if quantize:
        print("Applying post-training quantization...")
        
        if representative_dataset:
            print("Using full integer quantization with representative dataset")
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            converter.representative_dataset = representative_dataset
            converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
            converter.inference_input_type = tf.uint8
            converter.inference_output_type = tf.uint8
        else:
            print("Using default dynamic range quantization")
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
    
    # Convert the model
    print("Converting model to TFLite format...")
    try:
        tflite_model = converter.convert()
        
        # Save the model
        with open(output_path, 'wb') as f:
            f.write(tflite_model)
        
        print(f"Model successfully converted and saved to {output_path}")
        print(f"Model size: {os.path.getsize(output_path) / (1024 * 1024):.2f} MB")
        
        return True
    except Exception as e:
        print(f"Error converting model: {e}")
        return False

def main():
    """Main function to convert H5 model to TFLite format"""
    # Parse arguments
    args = parse_arguments()
    
    # Check if input model exists
    if not os.path.exists(args.input_model):
        print(f"Error: Input model file not found: {args.input_model}")
        sys.exit(1)
    
    # Get output path
    output_path = get_output_path(args)
    
    print(f"Converting model: {args.input_model} -> {output_path}")
    
    # Load Keras model
    model = load_keras_model(args.input_model)
    
    # Create representative dataset if specified
    representative_dataset = None
    if args.quantize and args.representative_dataset:
        representative_dataset = create_representative_dataset(args.representative_dataset)
    
    # Convert to TFLite
    success = convert_to_tflite(model, output_path, args.quantize, representative_dataset)
    
    if success:
        print("Conversion completed successfully!")
    else:
        print("Conversion failed.")
        sys.exit(1)

if __name__ == "__main__":
    main() 