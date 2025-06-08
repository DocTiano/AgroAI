#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
test_model.py - Test the trained plant disease detection model
Evaluates model performance on test dataset and generates a confusion matrix
"""

import os
import sys
import argparse
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd
import datetime

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Test plant disease detection model')
    parser.add_argument('--model-path', type=str, required=True,
                        help='Path to the trained model (H5 format)')
    parser.add_argument('--test-dir', type=str, required=True,
                        help='Path to the test dataset directory')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Directory to save test results (defaults to model directory)')
    parser.add_argument('--img-size', type=int, default=224,
                        help='Image size for model input (default: 224)')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size for testing (default: 32)')
    
    return parser.parse_args()

def get_output_dir(args):
    """Get the output directory for test results"""
    if args.output_dir:
        output_dir = args.output_dir
    else:
        output_dir = os.path.dirname(args.model_path)
    
    # Create a subdirectory for test results with timestamp
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    test_results_dir = os.path.join(output_dir, f'test_results_{timestamp}')
    os.makedirs(test_results_dir, exist_ok=True)
    
    return test_results_dir

def load_test_data(test_dir, img_size, batch_size):
    """Load test data using ImageDataGenerator"""
    test_datagen = ImageDataGenerator(rescale=1./255)
    
    test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=(img_size, img_size),
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False  # Important for correct class indices
    )
    
    return test_generator

def evaluate_model(model, test_generator):
    """Evaluate model on test data"""
    # Get evaluation metrics
    evaluation = model.evaluate(test_generator)
    
    # Generate predictions
    predictions = model.predict(test_generator)
    y_pred = np.argmax(predictions, axis=1)
    
    # Get true labels
    y_true = test_generator.classes
    
    # Get class names
    class_indices = test_generator.class_indices
    class_names = [k for k, v in sorted(class_indices.items(), key=lambda item: item[1])]
    
    return {
        'loss': evaluation[0],
        'accuracy': evaluation[1],
        'y_true': y_true,
        'y_pred': y_pred,
        'class_names': class_names,
        'predictions': predictions
    }

def generate_confusion_matrix(results, output_dir):
    """Generate and save confusion matrix"""
    cm = confusion_matrix(results['y_true'], results['y_pred'])
    plt.figure(figsize=(16, 14))
    
    # Normalize confusion matrix
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    # Plot confusion matrix
    sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=results['class_names'],
                yticklabels=results['class_names'])
    
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    
    # Save the plot
    cm_path = os.path.join(output_dir, 'confusion_matrix.png')
    plt.savefig(cm_path)
    plt.close()
    
    print(f"Confusion matrix saved to {cm_path}")
    
    return cm

def generate_classification_report(results, output_dir):
    """Generate and save classification report"""
    report = classification_report(
        results['y_true'],
        results['y_pred'],
        target_names=results['class_names'],
        output_dict=True
    )
    
    # Convert to DataFrame for better visualization
    report_df = pd.DataFrame(report).transpose()
    
    # Save as CSV
    report_path = os.path.join(output_dir, 'classification_report.csv')
    report_df.to_csv(report_path)
    
    # Also save as text
    with open(os.path.join(output_dir, 'classification_report.txt'), 'w') as f:
        f.write(classification_report(
            results['y_true'],
            results['y_pred'],
            target_names=results['class_names']
        ))
    
    print(f"Classification report saved to {report_path}")
    
    return report

def visualize_misclassified(test_generator, results, output_dir, max_samples=10):
    """Visualize some misclassified samples"""
    y_true = results['y_true']
    y_pred = results['y_pred']
    class_names = results['class_names']
    
    # Find misclassified samples
    misclassified_indices = np.where(y_true != y_pred)[0]
    
    if len(misclassified_indices) == 0:
        print("No misclassified samples found!")
        return
    
    # Select up to max_samples misclassified samples
    selected_indices = misclassified_indices[:min(max_samples, len(misclassified_indices))]
    
    # Create a figure to display misclassified samples
    fig, axes = plt.subplots(len(selected_indices), 1, figsize=(10, 4 * len(selected_indices)))
    
    # If only one sample, wrap axes in a list to make it iterable
    if len(selected_indices) == 1:
        axes = [axes]
    
    # Get the original test data
    test_data = []
    test_labels = []
    for i in range(len(test_generator)):
        batch_data, batch_labels = test_generator[i]
        test_data.append(batch_data)
        test_labels.append(batch_labels)
        if i+1 >= len(test_generator):
            break
    
    test_data = np.concatenate(test_data)
    
    # Plot each misclassified sample
    for i, idx in enumerate(selected_indices):
        ax = axes[i]
        img = test_data[idx]
        true_class = class_names[y_true[idx]]
        pred_class = class_names[y_pred[idx]]
        
        ax.imshow(img)
        ax.set_title(f"True: {true_class}\nPredicted: {pred_class}")
        ax.axis('off')
    
    plt.tight_layout()
    
    # Save the figure
    misclassified_path = os.path.join(output_dir, 'misclassified_samples.png')
    plt.savefig(misclassified_path)
    plt.close()
    
    print(f"Misclassified samples visualization saved to {misclassified_path}")

def generate_test_report(results, output_dir):
    """Generate a summary test report"""
    report_path = os.path.join(output_dir, 'test_report.md')
    
    with open(report_path, 'w') as f:
        f.write("# Plant Disease Detection Model Test Report\n\n")
        f.write(f"Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("## Performance Metrics\n\n")
        f.write(f"- **Loss**: {results['loss']:.4f}\n")
        f.write(f"- **Accuracy**: {results['accuracy']:.4f} ({results['accuracy']*100:.2f}%)\n\n")
        
        f.write("## Classification Details\n\n")
        f.write(f"- **Number of Classes**: {len(results['class_names'])}\n")
        f.write(f"- **Classes**: {', '.join(results['class_names'])}\n\n")
        
        f.write("## Files\n\n")
        f.write("- [Confusion Matrix](confusion_matrix.png)\n")
        f.write("- [Classification Report](classification_report.csv)\n")
        f.write("- [Misclassified Samples](misclassified_samples.png)\n\n")
        
        f.write("## Notes\n\n")
        f.write("- The confusion matrix shows the normalized prediction accuracy for each class\n")
        f.write("- The classification report provides precision, recall, and F1-score for each class\n")
        f.write("- A sample of misclassified images is provided for error analysis\n")
    
    print(f"Test report saved to {report_path}")

def main():
    """Main function to test the model"""
    # Parse arguments
    args = parse_arguments()
    
    # Check if model exists
    if not os.path.exists(args.model_path):
        print(f"Error: Model file not found: {args.model_path}")
        sys.exit(1)
    
    # Check if test directory exists
    if not os.path.exists(args.test_dir):
        print(f"Error: Test directory not found: {args.test_dir}")
        sys.exit(1)
    
    # Get output directory
    output_dir = get_output_dir(args)
    print(f"Test results will be saved to {output_dir}")
    
    # Load the model
    print(f"Loading model from {args.model_path}...")
    model = load_model(args.model_path)
    model.summary()
    
    # Load test data
    print(f"Loading test data from {args.test_dir}...")
    test_generator = load_test_data(args.test_dir, args.img_size, args.batch_size)
    print(f"Found {test_generator.samples} test samples in {len(test_generator.class_indices)} classes")
    
    # Evaluate model
    print("Evaluating model on test data...")
    results = evaluate_model(model, test_generator)
    print(f"Test Loss: {results['loss']:.4f}")
    print(f"Test Accuracy: {results['accuracy']:.4f} ({results['accuracy']*100:.2f}%)")
    
    # Generate confusion matrix
    print("Generating confusion matrix...")
    cm = generate_confusion_matrix(results, output_dir)
    
    # Generate classification report
    print("Generating classification report...")
    report = generate_classification_report(results, output_dir)
    
    # Visualize misclassified samples
    print("Visualizing misclassified samples...")
    visualize_misclassified(test_generator, results, output_dir)
    
    # Generate test report
    print("Generating test report...")
    generate_test_report(results, output_dir)
    
    print(f"Model testing completed! All results saved to {output_dir}")

if __name__ == "__main__":
    main() 