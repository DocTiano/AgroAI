"""
Plant Disease Detection - Transfer Learning Training Script
This script uses a pre-trained model (MobileNetV2) for improved plant disease detection.
"""
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import matplotlib.pyplot as plt
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TransferLearningTrainer:
    """
    Class for training plant disease detection models using transfer learning
    """
    def __init__(self, dataset_path, img_size=224, batch_size=32, base_model_name='MobileNetV2'):
        """
        Initialize the transfer learning trainer
        
        Args:
            dataset_path: Path to the dataset directory
            img_size: Input image size (default: 224x224)
            batch_size: Batch size for training (default: 32)
            base_model_name: Name of the pre-trained model to use (default: MobileNetV2)
        """
        self.dataset_path = dataset_path
        self.img_size = img_size
        self.batch_size = batch_size
        self.base_model_name = base_model_name
        self.model = None
        self.history = None
        self.class_names = []
        self.base_model = None
        
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
            preprocessing_function=tf.keras.applications.mobilenet_v2.preprocess_input,
            validation_split=validation_split,
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest'
        )
        
        # Validation data should only be preprocessed, not augmented
        valid_datagen = ImageDataGenerator(
            preprocessing_function=tf.keras.applications.mobilenet_v2.preprocess_input,
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
        Build a transfer learning model for plant disease classification
        
        Args:
            num_classes: Number of disease classes
            
        Returns:
            Compiled Keras model
        """
        logger.info(f"Building transfer learning model with {num_classes} output classes")
        
        # Load pre-trained base model without top layers
        base_model = MobileNetV2(
            weights='imagenet',
            include_top=False,
            input_shape=(self.img_size, self.img_size, 3)
        )
        
        # Freeze base model layers
        base_model.trainable = False
        
        # Add custom top layers
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(1024, activation='relu')(x)
        x = Dropout(0.5)(x)
        x = Dense(512, activation='relu')(x)
        x = Dropout(0.3)(x)
        predictions = Dense(num_classes, activation='softmax')(x)
        
        # Create the final model
        model = Model(inputs=base_model.input, outputs=predictions)
        
        # Save base model for later use in fine-tuning
        self.base_model = base_model
        
        # Compile the model
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # Print model summary
        model.summary()
        self.model = model
        return model
    
    def train_model(self, epochs=20, fine_tune=True, fine_tune_epochs=10):
        """
        Train the model on the prepared dataset
        
        Args:
            epochs: Number of training epochs for transfer learning phase
            fine_tune: Whether to fine-tune the base model after initial training
            fine_tune_epochs: Number of epochs for fine-tuning
            
        Returns:
            Training history
        """
        # Prepare dataset
        train_generator, validation_generator = self.prepare_dataset()
        
        # Get number of classes
        num_classes = len(train_generator.class_indices)
        
        # Build model
        logger.info("Building transfer learning model")
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
        
        # Initial training with frozen base model
        logger.info(f"Training top layers for {epochs} epochs")
        history1 = self.model.fit(
            train_generator,
            validation_data=validation_generator,
            epochs=epochs,
            callbacks=callbacks
        )
        
        # Fine-tuning phase
        if fine_tune and fine_tune_epochs > 0:
            logger.info("Fine-tuning the model")
            
            # Unfreeze some layers of the base model
            # Unfreeze the last 5 convolutional blocks of MobileNetV2
            for layer in self.base_model.layers[-30:]:
                layer.trainable = True
                
            # Recompile the model with a lower learning rate
            self.model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
                loss='categorical_crossentropy',
                metrics=['accuracy']
            )
            
            # Continue training with fine-tuning
            logger.info(f"Fine-tuning for {fine_tune_epochs} epochs")
            history2 = self.model.fit(
                train_generator,
                validation_data=validation_generator,
                epochs=fine_tune_epochs,
                callbacks=callbacks,
                initial_epoch=history1.epoch[-1] + 1
            )
            
            # Combine histories
            history = history1
            for k in history2.history:
                history.history[k].extend(history2.history[k])
            history.epoch.extend(history2.epoch)
        else:
            history = history1
        
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
        
        # Create the directory if it doesn't exist
        os.makedirs(save_path, exist_ok=True)
        
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
            
            # Enable optimizations
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            
            # Convert the model
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
            os.makedirs(save_path, exist_ok=True)
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
            test_datagen = ImageDataGenerator(
                preprocessing_function=tf.keras.applications.mobilenet_v2.preprocess_input
            )
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
    Example usage of the TransferLearningTrainer
    """
    # Set paths
    dataset_path = "path/to/dataset"  # Replace with your dataset path
    save_path = "path/to/save/model"  # Replace with your save path
    
    # Initialize trainer
    trainer = TransferLearningTrainer(dataset_path)
    
    # Train model (with fine-tuning)
    trainer.train_model(epochs=20, fine_tune=True, fine_tune_epochs=10)
    
    # Save model
    trainer.save_model(save_path)
    
    # Plot training history
    trainer.plot_training_history(save_path)
    
    # Evaluate model
    trainer.evaluate_model()


if __name__ == "__main__":
    main()


"""
HOW TO USE TRANSFER LEARNING FOR PLANT DISEASE DETECTION
=======================================================

This script provides an improved approach to training plant disease models using transfer learning.
Transfer learning leverages pre-trained models to achieve better results with less data and training time.

BENEFITS OF TRANSFER LEARNING:
-----------------------------
1. Higher accuracy with less training data
2. Faster convergence during training
3. Better generalization to new images
4. Works well even with a few hundred images per class

STEPS TO TRAIN YOUR MODEL:
------------------------

1. PREPARE YOUR DATASET
   -------------------
   Organize your dataset with the same structure:
   
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

2. INSTALL REQUIREMENTS
   ------------------
   Make sure you have:
   - tensorflow
   - numpy
   - matplotlib
   - pillow
   
   Note: This script requires more GPU memory than training from scratch.
   If you have memory issues, reduce the batch_size parameter.

3. RUN THE SCRIPT
   -----------
   Edit the main() function to set your paths, then run:
   
   ```
   python train_transfer_learning.py
   ```

4. UNDERSTANDING THE TRAINING PROCESS
   -------------------------------
   The training happens in two phases:
   
   a) Initial phase: Only the top layers are trained while the base MobileNetV2 model is frozen
   b) Fine-tuning phase: Some upper layers of the base model are unfrozen and trained with a lower learning rate

5. IMPROVE YOUR RESULTS
   -----------------
   - Increase the number of images per class (at least 100-200 per class is ideal)
   - Try different data augmentation settings
   - Experiment with different base models (ResNet50, EfficientNet)
   - Adjust learning rates and the number of layers to unfreeze during fine-tuning

After training, your model will be saved in both .h5 (Keras) and .tflite (TensorFlow Lite) formats,
ready to be used with the PlantDiseaseDetector class.
""" 