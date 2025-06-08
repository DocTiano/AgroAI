"""
Plant Disease Model Training Script
This script provides functionality to train a custom plant disease detection model
"""
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import matplotlib.pyplot as plt
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PlantDiseaseModelTrainer:
    """
    Class for training plant disease detection models
    """
    def __init__(self, dataset_path, img_size=224, batch_size=32):
        """
        Initialize the model trainer
        
        Args:
            dataset_path: Path to the dataset directory
            img_size: Input image size (default: 224x224)
            batch_size: Batch size for training (default: 32)
        """
        self.dataset_path = dataset_path
        self.img_size = img_size
        self.batch_size = batch_size
        self.model = None
        self.history = None
        self.class_names = []
        
    def prepare_dataset(self, validation_split=0.2):
        """
        Prepare the dataset for training
        
        Args:
            validation_split: Fraction of data to use for validation
            
        Returns:
            train_generator, validation_generator
        """
        logger.info(f"Preparing dataset from {self.dataset_path}")
        
        # Create data generators with augmentation for training
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            validation_split=validation_split,
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest'
        )
        
        # Validation data should only be rescaled, not augmented
        valid_datagen = ImageDataGenerator(
            rescale=1./255,
            validation_split=validation_split
        )
        
        # Training data generator
        train_generator = train_datagen.flow_from_directory(
            self.dataset_path,
            target_size=(self.img_size, self.img_size),
            batch_size=self.batch_size,
            class_mode='categorical',
            subset='training',
            shuffle=True
        )
        
        # Validation data generator
        validation_generator = valid_datagen.flow_from_directory(
            self.dataset_path,
            target_size=(self.img_size, self.img_size),
            batch_size=self.batch_size,
            class_mode='categorical',
            subset='validation',
            shuffle=False
        )
        
        # Store class names
        self.class_names = list(train_generator.class_indices.keys())
        logger.info(f"Found {len(self.class_names)} classes: {self.class_names}")
        
        return train_generator, validation_generator
    
    def build_model(self, num_classes):
        """
        Build a CNN model for plant disease classification
        
        Args:
            num_classes: Number of disease classes
            
        Returns:
            Compiled Keras model
        """
        logger.info(f"Building model with {num_classes} output classes")
        
        model = Sequential([
            # First Conv Block
            Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(self.img_size, self.img_size, 3)),
            MaxPooling2D(2, 2),
            
            # Second Conv Block
            Conv2D(64, (3, 3), activation='relu', padding='same'),
            MaxPooling2D(2, 2),
            
            # Third Conv Block
            Conv2D(128, (3, 3), activation='relu', padding='same'),
            MaxPooling2D(2, 2),
            
            # Fourth Conv Block
            Conv2D(256, (3, 3), activation='relu', padding='same'),
            MaxPooling2D(2, 2),
            
            # Flatten and Dense layers
            Flatten(),
            Dense(512, activation='relu'),
            Dropout(0.5),
            Dense(num_classes, activation='softmax')
        ])
        
        # Compile the model
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # Print model summary
        model.summary()
        self.model = model
        return model
    
    def train_model(self, epochs=20, use_pretrained=False, model_path=None):
        """
        Train the model on the prepared dataset
        
        Args:
            epochs: Number of training epochs
            use_pretrained: Whether to use a pretrained model
            model_path: Path to pretrained model weights (if use_pretrained is True)
            
        Returns:
            Training history
        """
        # Prepare dataset
        train_generator, validation_generator = self.prepare_dataset()
        
        # Get number of classes
        num_classes = len(train_generator.class_indices)
        
        # Build model or load pretrained
        if use_pretrained and model_path and os.path.exists(model_path):
            logger.info(f"Loading pretrained model from {model_path}")
            self.model = tf.keras.models.load_model(model_path)
        else:
            logger.info("Building new model")
            self.build_model(num_classes)
        
        # Define callbacks
        checkpoint_path = 'plant_disease_model_checkpoint.h5'
        callbacks = [
            ModelCheckpoint(
                checkpoint_path,
                monitor='val_accuracy',
                save_best_only=True,
                mode='max',
                verbose=1
            ),
            EarlyStopping(
                monitor='val_loss',
                patience=5,
                restore_best_weights=True,
                verbose=1
            )
        ]
        
        # Train the model
        logger.info(f"Training model for {epochs} epochs")
        history = self.model.fit(
            train_generator,
            validation_data=validation_generator,
            epochs=epochs,
            callbacks=callbacks
        )
        
        self.history = history
        return history
    
    def save_model(self, save_path, save_tflite=True):
        """
        Save the trained model
        
        Args:
            save_path: Path to save the model
            save_tflite: Whether to also save a TFLite version
        """
        if self.model is None:
            logger.error("No model to save. Train a model first.")
            return
        
        # Save Keras model
        model_path = os.path.join(save_path, 'plant_disease_model.h5')
        self.model.save(model_path)
        logger.info(f"Model saved to {model_path}")
        
        # Save class names
        class_names_path = os.path.join(save_path, 'class_names.txt')
        with open(class_names_path, 'w') as f:
            for class_name in self.class_names:
                f.write(f"{class_name}\n")
        logger.info(f"Class names saved to {class_names_path}")
        
        # Save as TFLite model if requested
        if save_tflite:
            # Convert to TFLite
            converter = tf.lite.TFLiteConverter.from_keras_model(self.model)
            tflite_model = converter.convert()
            
            # Save TFLite model
            tflite_path = os.path.join(save_path, 'plant_disease_model.tflite')
            with open(tflite_path, 'wb') as f:
                f.write(tflite_model)
            logger.info(f"TFLite model saved to {tflite_path}")
    
    def plot_training_history(self, save_path=None):
        """
        Plot training history
        
        Args:
            save_path: Optional path to save the plots
        """
        if self.history is None:
            logger.error("No training history. Train a model first.")
            return
        
        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Plot accuracy
        ax1.plot(self.history.history['accuracy'])
        ax1.plot(self.history.history['val_accuracy'])
        ax1.set_title('Model Accuracy')
        ax1.set_ylabel('Accuracy')
        ax1.set_xlabel('Epoch')
        ax1.legend(['Train', 'Validation'], loc='lower right')
        
        # Plot loss
        ax2.plot(self.history.history['loss'])
        ax2.plot(self.history.history['val_loss'])
        ax2.set_title('Model Loss')
        ax2.set_ylabel('Loss')
        ax2.set_xlabel('Epoch')
        ax2.legend(['Train', 'Validation'], loc='upper right')
        
        plt.tight_layout()
        
        # Save if path provided
        if save_path:
            plt.savefig(os.path.join(save_path, 'training_history.png'))
            logger.info(f"Training history plot saved to {save_path}")
        
        plt.show()
        
    def evaluate_model(self, test_data_path=None):
        """
        Evaluate the model on test data
        
        Args:
            test_data_path: Path to test data directory. If None, uses validation data.
            
        Returns:
            Evaluation metrics
        """
        if self.model is None:
            logger.error("No model to evaluate. Train a model first.")
            return
        
        # Create test data generator
        if test_data_path and os.path.exists(test_data_path):
            logger.info(f"Evaluating model on test data from {test_data_path}")
            test_datagen = ImageDataGenerator(rescale=1./255)
            test_generator = test_datagen.flow_from_directory(
                test_data_path,
                target_size=(self.img_size, self.img_size),
                batch_size=self.batch_size,
                class_mode='categorical',
                shuffle=False
            )
        else:
            # Use validation data if no test data provided
            logger.info("No test data provided. Using validation data.")
            _, test_generator = self.prepare_dataset()
        
        # Evaluate the model
        results = self.model.evaluate(test_generator)
        
        # Print results
        logger.info(f"Test Loss: {results[0]:.4f}")
        logger.info(f"Test Accuracy: {results[1]:.4f}")
        
        return results


def main():
    """
    Example usage of the PlantDiseaseModelTrainer
    """
    # Set paths
    dataset_path = "path/to/dataset"  # Replace with your dataset path
    save_path = "path/to/save/model"  # Replace with your save path
    
    # Create directories if they don't exist
    os.makedirs(save_path, exist_ok=True)
    
    # Initialize trainer
    trainer = PlantDiseaseModelTrainer(dataset_path)
    
    # Train model
    trainer.train_model(epochs=30)
    
    # Save model
    trainer.save_model(save_path)
    
    # Plot training history
    trainer.plot_training_history(save_path)
    
    # Evaluate model
    trainer.evaluate_model()


if __name__ == "__main__":
    main()


"""
HOW TO USE THIS SCRIPT
=====================

This script provides a comprehensive pipeline for training a plant disease detection model.
Follow these steps to train your own model:

1. PREPARE YOUR DATASET
   -------------------
   Your dataset should have the following structure:
   
   dataset/
   ├── disease_class_1/
   │   ├── image1.jpg
   │   ├── image2.jpg
   │   └── ...
   ├── disease_class_2/
   │   ├── image1.jpg
   │   ├── image2.jpg
   │   └── ...
   └── ...
   
   - Each disease should have its own folder
   - Images should be in JPG/PNG format
   - Recommended: at least 100 images per class for good results

2. INSTALL REQUIREMENTS
   ------------------
   Make sure you have the required packages:
   - tensorflow
   - numpy
   - matplotlib
   - pillow

3. CUSTOMIZE THE TRAINING
   -------------------
   Edit the main() function in this script:
   - Set dataset_path to your dataset location
   - Set save_path to where you want to save the model
   - Adjust epochs, batch size, etc. as needed

4. RUN THE SCRIPT
   -----------
   Run this script:
   
   ```
   python train_model.py
   ```

5. USE YOUR TRAINED MODEL
   -------------------
   After training, you'll have:
   - plant_disease_model.h5 (Keras model)
   - plant_disease_model.tflite (TensorFlow Lite model)
   - class_names.txt (List of disease classes)
   
   Update the PlantDiseaseDetector class to use your new model.

TIPS FOR BETTER RESULTS
----------------------
1. More diverse data = better model
2. Increase epochs (50-100) for better accuracy
3. Try different image augmentation parameters
4. Add more layers or use transfer learning for better results
5. Ensure class balance in your dataset
""" 