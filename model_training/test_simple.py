#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
test_simple.py - Simple script to test the plant disease detection model on a single image
"""

import os
import sys
import argparse
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import matplotlib.pyplot as plt
from PIL import Image
import cv2

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Test plant disease detection model on a single image')
    parser.add_argument('--model-path', type=str, required=True,
                        help='Path to the trained model (H5 format)')
    parser.add_argument('--image-path', type=str, required=True,
                        help='Path to the image to test')
    parser.add_argument('--class-names-path', type=str, default=None,
                        help='Path to the class names file (defaults to same directory as model)')
    parser.add_argument('--img-size', type=int, default=224,
                        help='Image size for model input (default: 224)')
    parser.add_argument('--visualize', action='store_true',
                        help='Visualize the image and prediction')
    parser.add_argument('--save-output', type=str, default=None,
                        help='Path to save the visualization output')
    
    return parser.parse_args()

def load_class_names(model_path, class_names_path=None):
    """Load class names from a text file"""
    if class_names_path is None:
        # Assume class_names.txt is in the same directory as the model
        model_dir = os.path.dirname(model_path)
        class_names_path = os.path.join(model_dir, 'class_names.txt')
    
    if not os.path.exists(class_names_path):
        print(f"Error: Class names file not found at {class_names_path}")
        sys.exit(1)
    
    with open(class_names_path, 'r') as f:
        class_names = [line.strip() for line in f.readlines()]
    
    return class_names

def preprocess_image(image_path, img_size):
    """Preprocess the image for model input"""
    try:
        # Load and resize image
        img = load_img(image_path, target_size=(img_size, img_size))
        
        # Convert to array and normalize
        img_array = img_to_array(img)
        img_array = img_array / 255.0
        
        # Expand dimensions to match model input shape
        img_array = np.expand_dims(img_array, axis=0)
        
        return img_array, img
    except Exception as e:
        print(f"Error preprocessing image: {e}")
        sys.exit(1)

def predict_disease(model, img_array, class_names):
    """Make a prediction using the model"""
    # Make prediction
    predictions = model.predict(img_array)
    
    # Get the predicted class and confidence
    predicted_class_idx = np.argmax(predictions[0])
    confidence = predictions[0][predicted_class_idx]
    
    # Get the class name
    predicted_class = class_names[predicted_class_idx]
    
    # Get top 3 predictions
    top_indices = np.argsort(predictions[0])[-3:][::-1]
    top_predictions = [
        (class_names[idx], predictions[0][idx])
        for idx in top_indices
    ]
    
    return {
        'class_idx': predicted_class_idx,
        'class_name': predicted_class,
        'confidence': confidence,
        'top_predictions': top_predictions,
        'raw_predictions': predictions[0]
    }

def visualize_prediction(img, prediction, save_path=None):
    """Visualize the image and prediction"""
    # Create a figure
    plt.figure(figsize=(12, 8))
    
    # Plot the image
    plt.subplot(1, 2, 1)
    plt.imshow(img)
    plt.title('Input Image')
    plt.axis('off')
    
    # Plot the top 3 predictions
    plt.subplot(1, 2, 2)
    top_predictions = prediction['top_predictions']
    class_names = [p[0] for p in top_predictions]
    confidence = [p[1] for p in top_predictions]
    
    # Create a horizontal bar chart
    y_pos = np.arange(len(class_names))
    plt.barh(y_pos, confidence, align='center')
    plt.yticks(y_pos, class_names)
    plt.xlabel('Confidence')
    plt.title('Top Predictions')
    
    # Adjust layout
    plt.tight_layout()
    
    # Save the visualization if requested
    if save_path:
        plt.savefig(save_path)
        print(f"Visualization saved to {save_path}")
    
    # Show the plot
    plt.show()

def main():
    """Main function to test the model on a single image"""
    # Parse arguments
    args = parse_arguments()
    
    # Check if model exists
    if not os.path.exists(args.model_path):
        print(f"Error: Model file not found: {args.model_path}")
        sys.exit(1)
    
    # Check if image exists
    if not os.path.exists(args.image_path):
        print(f"Error: Image file not found: {args.image_path}")
        sys.exit(1)
    
    # Load class names
    class_names = load_class_names(args.model_path, args.class_names_path)
    print(f"Loaded {len(class_names)} class names")
    
    # Load the model
    print(f"Loading model from {args.model_path}...")
    model = load_model(args.model_path)
    
    # Preprocess the image
    print(f"Processing image: {args.image_path}")
    img_array, img = preprocess_image(args.image_path, args.img_size)
    
    # Make a prediction
    print("Making prediction...")
    prediction = predict_disease(model, img_array, class_names)
    
    # Print the results
    print("\nPrediction Results:")
    print(f"Predicted Class: {prediction['class_name']}")
    print(f"Confidence: {prediction['confidence']:.4f} ({prediction['confidence']*100:.2f}%)")
    
    print("\nTop 3 Predictions:")
    for i, (class_name, conf) in enumerate(prediction['top_predictions']):
        print(f"{i+1}. {class_name}: {conf:.4f} ({conf*100:.2f}%)")
    
    # Visualize the prediction if requested
    if args.visualize or args.save_output:
        visualize_prediction(img, prediction, args.save_output)

if __name__ == "__main__":
    main() 