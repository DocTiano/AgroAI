"""
Image Processing Utilities for AgroAI
This module provides functions for preprocessing images before feeding them to the CNN model
"""
import cv2
import numpy as np
from PIL import Image
import os

def preprocess_image(image_path, target_size=(224, 224)):
    """
    Preprocess an image for the CNN model
    
    Args:
        image_path (str): Path to the image file
        target_size (tuple): Target size for the image (height, width)
        
    Returns:
        numpy.ndarray: Preprocessed image ready for model input
    """
    # Check if file exists
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")
    
    # Read image
    try:
        # Try with PIL first (handles more formats)
        img = Image.open(image_path)
        img = img.convert('RGB')  # Ensure RGB format
        img = img.resize(target_size)
        img_array = np.array(img)
    except Exception as e:
        # Fallback to OpenCV
        print(f"PIL processing failed, using OpenCV: {e}")
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Failed to load image: {image_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
        img = cv2.resize(img, target_size)
        img_array = img
    
    # Normalize pixel values to [0, 1]
    img_array = img_array.astype(np.float32) / 255.0
    
    return img_array

def enhance_image(image_path, output_path=None):
    """
    Enhance image quality for better disease detection
    
    Args:
        image_path (str): Path to the input image
        output_path (str, optional): Path to save the enhanced image
        
    Returns:
        numpy.ndarray: Enhanced image
    """
    # Read image
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Failed to load image: {image_path}")
    
    # Convert to LAB color space
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    
    # Split LAB channels
    l, a, b = cv2.split(lab)
    
    # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) to L channel
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    
    # Merge enhanced L channel with original A and B channels
    enhanced_lab = cv2.merge((cl, a, b))
    
    # Convert back to BGR color space
    enhanced_img = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
    
    # Apply slight sharpening
    kernel = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]])
    enhanced_img = cv2.filter2D(enhanced_img, -1, kernel)
    
    # Save enhanced image if output path is provided
    if output_path:
        cv2.imwrite(output_path, enhanced_img)
    
    return enhanced_img

def segment_leaf(image_path, output_path=None):
    """
    Segment the leaf from the background
    
    Args:
        image_path (str): Path to the input image
        output_path (str, optional): Path to save the segmented image
        
    Returns:
        numpy.ndarray: Segmented image with isolated leaf
    """
    # Read image
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Failed to load image: {image_path}")
    
    # Convert to HSV color space
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # Define range for green color detection
    lower_green = np.array([25, 40, 50])
    upper_green = np.array([85, 255, 255])
    
    # Create mask for green areas
    mask = cv2.inRange(hsv, lower_green, upper_green)
    
    # Apply morphological operations to improve mask
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    
    # Find contours in the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # If no contours found, return original image
    if not contours:
        return img
    
    # Find the largest contour (assuming it's the leaf)
    largest_contour = max(contours, key=cv2.contourArea)
    
    # Create a mask with only the largest contour
    leaf_mask = np.zeros_like(mask)
    cv2.drawContours(leaf_mask, [largest_contour], 0, 255, -1)
    
    # Apply the mask to the original image
    segmented = cv2.bitwise_and(img, img, mask=leaf_mask)
    
    # Save segmented image if output path is provided
    if output_path:
        cv2.imwrite(output_path, segmented)
    
    return segmented

def extract_features(image):
    """
    Extract features from the image that might be useful for disease detection
    
    Args:
        image (numpy.ndarray): Input image
        
    Returns:
        dict: Dictionary of extracted features
    """
    # Convert to HSV color space
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Split channels
    h, s, v = cv2.split(hsv)
    
    # Calculate color histograms
    h_hist = cv2.calcHist([h], [0], None, [30], [0, 180])
    s_hist = cv2.calcHist([s], [0], None, [32], [0, 256])
    v_hist = cv2.calcHist([v], [0], None, [32], [0, 256])
    
    # Normalize histograms
    h_hist = cv2.normalize(h_hist, h_hist, 0, 1, cv2.NORM_MINMAX)
    s_hist = cv2.normalize(s_hist, s_hist, 0, 1, cv2.NORM_MINMAX)
    v_hist = cv2.normalize(v_hist, v_hist, 0, 1, cv2.NORM_MINMAX)
    
    # Calculate texture features using GLCM (Gray-Level Co-occurrence Matrix)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Calculate average color values
    avg_color_per_channel = np.mean(image, axis=(0, 1))
    
    # Calculate standard deviation of color values
    std_color_per_channel = np.std(image, axis=(0, 1))
    
    # Compile features
    features = {
        'avg_blue': float(avg_color_per_channel[0]),
        'avg_green': float(avg_color_per_channel[1]),
        'avg_red': float(avg_color_per_channel[2]),
        'std_blue': float(std_color_per_channel[0]),
        'std_green': float(std_color_per_channel[1]),
        'std_red': float(std_color_per_channel[2]),
        'h_hist': h_hist.flatten().tolist(),
        's_hist': s_hist.flatten().tolist(),
        'v_hist': v_hist.flatten().tolist(),
    }
    
    return features

def data_augmentation(image):
    """
    Apply data augmentation to an image
    
    Args:
        image (numpy.ndarray): Input image
        
    Returns:
        list: List of augmented images
    """
    augmented_images = []
    
    # Original image
    augmented_images.append(image)
    
    # Horizontal flip
    h_flip = cv2.flip(image, 1)
    augmented_images.append(h_flip)
    
    # Vertical flip
    v_flip = cv2.flip(image, 0)
    augmented_images.append(v_flip)
    
    # Rotation (90 degrees)
    rows, cols = image.shape[:2]
    rotation_matrix = cv2.getRotationMatrix2D((cols/2, rows/2), 90, 1)
    rotated_90 = cv2.warpAffine(image, rotation_matrix, (cols, rows))
    augmented_images.append(rotated_90)
    
    # Brightness adjustment (increase)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    v = cv2.add(v, 30)
    v[v > 255] = 255
    hsv_bright = cv2.merge((h, s, v))
    bright = cv2.cvtColor(hsv_bright, cv2.COLOR_HSV2BGR)
    augmented_images.append(bright)
    
    # Brightness adjustment (decrease)
    v = cv2.subtract(v, 60)
    v[v < 0] = 0
    hsv_dark = cv2.merge((h, s, v))
    dark = cv2.cvtColor(hsv_dark, cv2.COLOR_HSV2BGR)
    augmented_images.append(dark)
    
    return augmented_images
