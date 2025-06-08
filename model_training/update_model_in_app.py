#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
update_model_in_app.py - Update trained model in the application
This script copies the trained model to the app directory and updates app configurations
"""

import os
import sys
import shutil
import argparse
import json
import datetime

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Update the model in the application')
    parser.add_argument('--model-dir', type=str, required=True,
                        help='Directory containing the trained model files')
    parser.add_argument('--app-dir', type=str, default=None,
                        help='Application directory (defaults to the app directory in project root)')
    parser.add_argument('--model-name', type=str, default='plant_disease_model.tflite',
                        help='Name of the TFLite model file (default: plant_disease_model.tflite)')
    parser.add_argument('--backup', action='store_true',
                        help='Create a backup of the existing model before replacing')
    
    return parser.parse_args()

def get_app_dir(args):
    """Get the application directory"""
    if args.app_dir:
        return args.app_dir
    
    # Default to app directory in the project root
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base_dir, 'app')

def backup_existing_model(app_dir, model_name):
    """Create a backup of the existing model"""
    model_path = os.path.join(app_dir, 'models', model_name)
    
    # Check if model exists
    if not os.path.exists(model_path):
        print(f"No existing model found at {model_path}. Skipping backup.")
        return
    
    # Create backup directory if it doesn't exist
    backup_dir = os.path.join(app_dir, 'models', 'backups')
    os.makedirs(backup_dir, exist_ok=True)
    
    # Backup the model with timestamp
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    backup_path = os.path.join(backup_dir, f"{os.path.splitext(model_name)[0]}_{timestamp}.tflite")
    
    # Copy the model to backup location
    shutil.copy2(model_path, backup_path)
    print(f"Backed up existing model to {backup_path}")

def copy_model_files(model_dir, app_dir, model_name):
    """Copy model files to the application directory"""
    # Ensure app models directory exists
    app_models_dir = os.path.join(app_dir, 'models')
    os.makedirs(app_models_dir, exist_ok=True)
    
    # Copy TFLite model
    tflite_model_src = os.path.join(model_dir, model_name)
    tflite_model_dst = os.path.join(app_models_dir, model_name)
    
    if not os.path.exists(tflite_model_src):
        raise FileNotFoundError(f"Model file not found: {tflite_model_src}")
    
    shutil.copy2(tflite_model_src, tflite_model_dst)
    print(f"Copied model: {tflite_model_src} -> {tflite_model_dst}")
    
    # Copy class names file if it exists
    class_names_src = os.path.join(model_dir, 'class_names.txt')
    class_names_dst = os.path.join(app_models_dir, 'class_names.txt')
    
    if os.path.exists(class_names_src):
        shutil.copy2(class_names_src, class_names_dst)
        print(f"Copied class names: {class_names_src} -> {class_names_dst}")
    else:
        print(f"Warning: Class names file not found at {class_names_src}")

def update_config_file(app_dir, model_name):
    """Update the application configuration file with the new model information"""
    config_path = os.path.join(app_dir, 'config.json')
    
    # Create default config if it doesn't exist
    if not os.path.exists(config_path):
        config = {
            "model": {
                "name": model_name,
                "updated_at": datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                "input_shape": [1, 224, 224, 3],
                "normalize": True,
                "class_names_file": "class_names.txt"
            }
        }
    else:
        # Load existing config
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        # Update model information
        if "model" not in config:
            config["model"] = {}
        
        config["model"]["name"] = model_name
        config["model"]["updated_at"] = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    # Save updated config
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=4)
    
    print(f"Updated configuration at {config_path}")

def main():
    """Main function to update the model in the application"""
    # Parse arguments
    args = parse_arguments()
    
    # Get application directory
    app_dir = get_app_dir(args)
    
    # Check if application directory exists
    if not os.path.exists(app_dir):
        print(f"Error: Application directory not found: {app_dir}")
        sys.exit(1)
    
    print(f"Updating model in application directory: {app_dir}")
    print(f"Using model from: {args.model_dir}")
    
    # Backup existing model if requested
    if args.backup:
        backup_existing_model(app_dir, args.model_name)
    
    # Copy model files
    copy_model_files(args.model_dir, app_dir, args.model_name)
    
    # Update configuration file
    update_config_file(app_dir, args.model_name)
    
    print("Model update complete!")

if __name__ == "__main__":
    main() 