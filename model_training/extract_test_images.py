#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
extract_test_images.py - Extract a subset of images from the dataset for testing
This script splits images from the training set into a separate test set
"""

import os
import sys
import argparse
import random
import shutil
from pathlib import Path
import math

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Extract test images from dataset')
    parser.add_argument('--dataset-dir', type=str, required=True,
                        help='Path to the dataset directory')
    parser.add_argument('--test-size', type=float, default=0.2,
                        help='Proportion of images to extract for testing (default: 0.2)')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Output directory for test images (default: dataset_dir/test)')
    parser.add_argument('--random-seed', type=int, default=42,
                        help='Random seed for reproducibility (default: 42)')
    parser.add_argument('--min-samples', type=int, default=10,
                        help='Minimum number of samples per class in test set (default: 10)')
    
    return parser.parse_args()

def setup_directories(dataset_dir, output_dir=None):
    """Set up directories for the test set"""
    # If output_dir is not specified, create a test directory in the dataset directory
    if output_dir is None:
        output_dir = os.path.join(dataset_dir, 'test')
    
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Get the train directory
    train_dir = os.path.join(dataset_dir, 'train')
    if not os.path.exists(train_dir):
        # If there's no train subdirectory, assume the dataset_dir is already the train directory
        train_dir = dataset_dir
    
    return train_dir, output_dir

def extract_test_images(train_dir, test_dir, test_size=0.2, random_seed=42, min_samples=10):
    """Extract images from the training set to create a test set"""
    random.seed(random_seed)
    
    # Get all class directories
    class_dirs = [d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))]
    
    # Total statistics
    total_classes = len(class_dirs)
    total_train_images = 0
    total_test_images = 0
    
    print(f"Found {total_classes} classes in the training directory")
    
    # Process each class
    for class_name in class_dirs:
        class_dir = os.path.join(train_dir, class_name)
        
        # Create the corresponding test class directory
        test_class_dir = os.path.join(test_dir, class_name)
        os.makedirs(test_class_dir, exist_ok=True)
        
        # Get all image files in the class directory
        image_files = [f for f in os.listdir(class_dir) 
                      if os.path.isfile(os.path.join(class_dir, f)) 
                      and f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        # Calculate the number of images to extract
        num_images = len(image_files)
        num_test = max(min_samples, math.ceil(num_images * test_size))
        
        # Make sure we don't try to extract more images than available
        num_test = min(num_test, num_images - min_samples)
        
        # If the class has too few images, skip it
        if num_images < 2 * min_samples:
            print(f"  Skipping {class_name}: only {num_images} images (need at least {2 * min_samples})")
            continue
        
        # Randomly select images for the test set
        test_images = random.sample(image_files, num_test)
        
        # Copy the selected images to the test directory
        for image in test_images:
            src_path = os.path.join(class_dir, image)
            dst_path = os.path.join(test_class_dir, image)
            shutil.copy2(src_path, dst_path)
        
        print(f"  {class_name}: {num_test}/{num_images} images extracted to test set")
        
        # Update statistics
        total_train_images += num_images
        total_test_images += num_test
    
    # Print overall statistics
    print("\nExtraction completed:")
    print(f"- Total classes: {total_classes}")
    print(f"- Total training images: {total_train_images}")
    print(f"- Total test images: {total_test_images}")
    print(f"- Test ratio: {total_test_images / (total_train_images + total_test_images):.2f}")

def main():
    """Main function to extract test images"""
    # Parse arguments
    args = parse_arguments()
    
    # Set up directories
    train_dir, test_dir = setup_directories(args.dataset_dir, args.output_dir)
    
    print(f"Extracting test images from {train_dir} to {test_dir}")
    print(f"Test size: {args.test_size}")
    
    # Extract test images
    extract_test_images(
        train_dir,
        test_dir,
        test_size=args.test_size,
        random_seed=args.random_seed,
        min_samples=args.min_samples
    )
    
    print(f"\nTest set creation completed. Test images saved to {test_dir}")

if __name__ == "__main__":
    main() 