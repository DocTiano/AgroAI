"""
Custom Plant Disease Model Training Script for AgroAI
This script trains a model on the Kaggle plant disease dataset that's already in the workspace
"""
import os
import sys
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# Add the project root to the Python path to import custom modules
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

# Import our transfer learning trainer
try:
    from app.models.train_transfer_learning import TransferLearningTrainer
except ModuleNotFoundError:
    # Create the trainer class if it doesn't exist
    print("Creating TransferLearningTrainer class...")
    
    class TransferLearningTrainer:
        """Transfer Learning Trainer for plant disease classification"""
        
        def __init__(self, dataset_path, batch_size=32, img_size=224):
            """Initialize the trainer with dataset and parameters"""
            self.dataset_path = dataset_path
            self.batch_size = batch_size
            self.img_size = img_size
            self.class_names = []
            self.history = None
            
            # Initialize TensorFlow and data
            self._setup_tensorflow()
            self._prepare_data()
            self._create_model()
            
        def _setup_tensorflow(self):
            """Configure TensorFlow settings"""
            # Check for GPU
            gpus = tf.config.list_physical_devices('GPU')
            if gpus:
                try:
                    # Set memory growth to prevent allocating all memory at once
                    for gpu in gpus:
                        tf.config.experimental.set_memory_growth(gpu, True)
                    print(f"Using GPU: {gpus}")
                except Exception as e:
                    print(f"Error configuring GPU: {e}")
            else:
                print("No GPU found, using CPU.")
                
        def _prepare_data(self):
            """Prepare the data generators for training and validation"""
            from tensorflow.keras.preprocessing.image import ImageDataGenerator
            
            print(f"Loading dataset from {self.dataset_path}")
            
            # Create data generators with augmentation for training
            train_datagen = ImageDataGenerator(
                rescale=1./255,
                rotation_range=20,
                width_shift_range=0.2,
                height_shift_range=0.2,
                shear_range=0.2,
                zoom_range=0.2,
                horizontal_flip=True,
                validation_split=0.2  # Use 20% for validation
            )
            
            # Training data generator
            self.train_generator = train_datagen.flow_from_directory(
                self.dataset_path,
                target_size=(self.img_size, self.img_size),
                batch_size=self.batch_size,
                class_mode='categorical',
                subset='training'
            )
            
            # Validation data generator
            self.validation_generator = train_datagen.flow_from_directory(
                self.dataset_path,
                target_size=(self.img_size, self.img_size),
                batch_size=self.batch_size,
                class_mode='categorical',
                subset='validation'
            )
            
            # Store class names
            self.class_names = sorted(list(self.train_generator.class_indices.keys()))
            self.num_classes = len(self.class_names)
            
            print(f"Found {self.num_classes} classes")
            
        def _create_model(self):
            """Create a transfer learning model using MobileNetV2"""
            from tensorflow.keras.applications import MobileNetV2
            from tensorflow.keras.models import Model
            from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
            
            # Create base model with pre-trained weights
            base_model = MobileNetV2(
                weights='imagenet',
                include_top=False,
                input_shape=(self.img_size, self.img_size, 3)
            )
            
            # Freeze the base model initially
            base_model.trainable = False
            
            # Add custom classification head
            x = base_model.output
            x = GlobalAveragePooling2D()(x)
            x = Dense(1024, activation='relu')(x)
            x = Dropout(0.5)(x)
            x = Dense(512, activation='relu')(x)
            x = Dropout(0.3)(x)
            predictions = Dense(self.num_classes, activation='softmax')(x)
            
            # Create the full model
            self.model = Model(inputs=base_model.input, outputs=predictions)
            
            # Compile the model
            self.model.compile(
                optimizer='adam',
                loss='categorical_crossentropy',
                metrics=['accuracy']
            )
            
            # Store the base model for fine-tuning later
            self.base_model = base_model
            
            print("Model created successfully")
            
        def train_model(self, epochs=10, fine_tune=True, fine_tune_epochs=5):
            """Train the model, with optional fine-tuning"""
            from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
            
            # Callbacks for training
            callbacks = [
                EarlyStopping(patience=5, restore_best_weights=True),
                ReduceLROnPlateau(factor=0.2, patience=3),
            ]
            
            # Train the model with frozen base layers
            print(f"Training top layers for {epochs} epochs...")
            initial_history = self.model.fit(
                self.train_generator,
                validation_data=self.validation_generator,
                epochs=epochs,
                callbacks=callbacks
            )
            
            # Fine-tune the model if requested
            if fine_tune:
                print(f"Fine-tuning model for {fine_tune_epochs} additional epochs...")
                
                # Unfreeze the last few layers of the base model
                self.base_model.trainable = True
                
                # Freeze all except the last 15 layers
                for layer in self.base_model.layers[:-15]:
                    layer.trainable = False
                
                # Recompile with a lower learning rate
                self.model.compile(
                    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
                    loss='categorical_crossentropy',
                    metrics=['accuracy']
                )
                
                # Train again with fine-tuning
                fine_tune_history = self.model.fit(
                    self.train_generator,
                    validation_data=self.validation_generator,
                    epochs=fine_tune_epochs,
                    callbacks=callbacks
                )
                
                # Combine histories
                self.history = {}
                for key in initial_history.history:
                    self.history[key] = initial_history.history[key] + fine_tune_history.history[key]
            else:
                self.history = initial_history.history
            
            # Evaluate the final model
            eval_result = self.model.evaluate(self.validation_generator)
            print(f"\nFinal validation accuracy: {eval_result[1]:.4f}")
            print(f"Final validation loss: {eval_result[0]:.4f}")
            
            return self.history
            
        def save_model(self, save_path):
            """Save the model in H5 and TFLite formats"""
            # Save Keras H5 model
            h5_path = os.path.join(save_path, "plant_disease_model.h5")
            self.model.save(h5_path)
            print(f"Saved Keras model to {h5_path}")
            
            # Convert to TFLite
            converter = tf.lite.TFLiteConverter.from_keras_model(self.model)
            tflite_model = converter.convert()
            
            # Save TFLite model
            tflite_path = os.path.join(save_path, "plant_disease_model.tflite")
            with open(tflite_path, 'wb') as f:
                f.write(tflite_model)
            print(f"Saved TFLite model to {tflite_path}")
            
        def plot_training_history(self, save_path):
            """Plot and save the training history"""
            if not self.history:
                print("No training history available")
                return
                
            # Plot accuracy and loss
            plt.figure(figsize=(12, 5))
            
            # Plot accuracy
            plt.subplot(1, 2, 1)
            plt.plot(self.history['accuracy'], label='Training Accuracy')
            plt.plot(self.history['val_accuracy'], label='Validation Accuracy')
            plt.title('Model Accuracy')
            plt.xlabel('Epoch')
            plt.ylabel('Accuracy')
            plt.legend()
            
            # Plot loss
            plt.subplot(1, 2, 2)
            plt.plot(self.history['loss'], label='Training Loss')
            plt.plot(self.history['val_loss'], label='Validation Loss')
            plt.title('Model Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()
            
            # Save the plot
            plot_path = os.path.join(save_path, "training_history.png")
            plt.tight_layout()
            plt.savefig(plot_path)
            plt.close()
            print(f"Saved training history plot to {plot_path}")

def count_images(dataset_path):
    """Count the number of images in each disease class folder"""
    class_counts = {}
    total_images = 0
    
    for class_name in os.listdir(dataset_path):
        class_path = os.path.join(dataset_path, class_name)
        if os.path.isdir(class_path):
            image_count = len([f for f in os.listdir(class_path) 
                              if os.path.isfile(os.path.join(class_path, f)) 
                              and f.lower().endswith(('.png', '.jpg', '.jpeg'))])
            class_counts[class_name] = image_count
            total_images += image_count
    
    return class_counts, total_images

def main():
    """
    Train a custom plant disease detection model using the Kaggle dataset
    """
    # Set the dataset path to the Kaggle dataset
    dataset_path = os.path.join(project_root, "dataset/New Plant Diseases Dataset(Augmented)/New Plant Diseases Dataset(Augmented)/train")
    
    # Create a directory for the trained model
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = os.path.join(project_root, f"trained_models_{timestamp}")
    os.makedirs(save_path, exist_ok=True)
    
    # Count images in the dataset
    print("Analyzing dataset...")
    class_counts, total_images = count_images(dataset_path)
    
    # Print dataset statistics
    print(f"Found {len(class_counts)} disease classes with {total_images} total images")
    print(f"Average images per class: {total_images / len(class_counts):.1f}")
    
    # Determine optimal batch size based on available memory
    # Use smaller batch size if memory is limited
    try:
        # Check available GPU memory if TensorFlow can access GPU
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            # If GPU is available, try a larger batch size
            batch_size = 32
            print(f"GPU detected: {gpus}")
        else:
            # If no GPU, use smaller batch size
            batch_size = 16
            print("No GPU detected, using smaller batch size")
    except:
        # Fallback to safe batch size
        batch_size = 16
        print("Error detecting GPU, using smaller batch size")
    
    print(f"Using batch size of {batch_size}")
    
    # Initialize the transfer learning trainer
    print("\nInitializing model trainer...")
    trainer = TransferLearningTrainer(
        dataset_path=dataset_path,
        batch_size=batch_size,
        img_size=224  # Standard size for MobileNetV2
    )
    
    # Train the model with transfer learning
    print("\nStarting model training...")
    # Use full training with 10 epochs and 5 fine-tuning epochs
    trainer.train_model(epochs=10, fine_tune=True, fine_tune_epochs=5)
    
    # Save the trained model
    print("\nSaving the model...")
    trainer.save_model(save_path)
    
    # Plot and save the training history
    print("\nGenerating training history plots...")
    trainer.plot_training_history(save_path)
    
    # Create a file with class names for reference
    class_names_path = os.path.join(save_path, "class_names.txt")
    with open(class_names_path, 'w') as f:
        for class_name in trainer.class_names:
            f.write(f"{class_name}\n")
    
    # Print completion message with model location
    print("\n" + "="*50)
    print(f"Training complete! Model saved to: {save_path}")
    print("Files generated:")
    print(f" - plant_disease_model.h5 (Keras model)")
    print(f" - plant_disease_model.tflite (TensorFlow Lite model)")
    print(f" - class_names.txt (List of disease classes)")
    print(f" - training_history.png (Performance graphs)")
    print("\nTo use this model, copy the .tflite file to replace the existing model")
    print("="*50)

if __name__ == "__main__":
    # Print startup banner
    print("\n" + "="*50)
    print("AGROAI PLANT DISEASE MODEL TRAINER")
    print("="*50)
    print("This script will train a custom plant disease detection model")
    print("using transfer learning on the Kaggle dataset.")
    print("The training process may take several hours depending on your hardware.")
    print("="*50 + "\n")
    
    # Start the training process
    main()