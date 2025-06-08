"""
Disease Detection Model using Convolutional Neural Networks
This module implements a CNN-based model for crop disease detection
"""
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.applications import ResNet50, VGG16
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.optimizers import Adam

class DiseaseModel:
    """
    CNN-based model for crop disease detection
    Uses transfer learning with pre-trained models (ResNet50 or VGG16)
    """
    
    def __init__(self, model_path=None, model_type='resnet50', num_classes=38, input_shape=(224, 224, 3)):
        """
        Initialize the disease detection model
        
        Args:
            model_path (str): Path to a saved model file. If provided, loads the model from file.
            model_type (str): Type of base model to use ('resnet50' or 'vgg16')
            num_classes (int): Number of disease classes to predict
            input_shape (tuple): Input image dimensions (height, width, channels)
        """
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model_type = model_type
        
        # Disease class mapping (would be loaded from a file in a real implementation)
        self.class_names = {
            0: "Healthy",
            1: "Apple___Apple_scab",
            2: "Apple___Black_rot",
            # ... other classes would be defined here
        }
        
        # Load or build model
        if model_path and os.path.exists(model_path):
            self.model = load_model(model_path)
            print(f"Model loaded from {model_path}")
        else:
            self.model = self._build_model()
            print(f"New model created with {model_type} base")
    
    def _build_model(self):
        """
        Build the CNN model using transfer learning
        
        Returns:
            Model: Compiled Keras model
        """
        # Create base model with pre-trained weights
        if self.model_type.lower() == 'resnet50':
            base_model = ResNet50(weights='imagenet', include_top=False, input_shape=self.input_shape)
        elif self.model_type.lower() == 'vgg16':
            base_model = VGG16(weights='imagenet', include_top=False, input_shape=self.input_shape)
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}. Use 'resnet50' or 'vgg16'.")
        
        # Add custom classification layers
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(1024, activation='relu')(x)
        x = Dropout(0.5)(x)
        x = Dense(512, activation='relu')(x)
        x = Dropout(0.3)(x)
        predictions = Dense(self.num_classes, activation='softmax')(x)
        
        # Create the model
        model = Model(inputs=base_model.input, outputs=predictions)
        
        # Freeze the base model layers
        for layer in base_model.layers:
            layer.trainable = False
        
        # Compile the model
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def train(self, train_data, validation_data, epochs=10, batch_size=32, unfreeze_layers=0):
        """
        Train the model on the provided data
        
        Args:
            train_data: Training data generator or dataset
            validation_data: Validation data generator or dataset
            epochs (int): Number of training epochs
            batch_size (int): Batch size for training
            unfreeze_layers (int): Number of layers to unfreeze from the base model for fine-tuning
        
        Returns:
            History object with training metrics
        """
        # Unfreeze some layers for fine-tuning if specified
        if unfreeze_layers > 0:
            # Get the base model (first part of the model)
            base_model = self.model.layers[0]
            
            # Calculate how many layers to unfreeze
            trainable_layers = min(unfreeze_layers, len(base_model.layers))
            
            # Unfreeze the last 'trainable_layers' layers
            for layer in base_model.layers[-trainable_layers:]:
                layer.trainable = True
            
            # Recompile the model with a lower learning rate for fine-tuning
            self.model.compile(
                optimizer=Adam(learning_rate=0.0001),
                loss='categorical_crossentropy',
                metrics=['accuracy']
            )
        
        # Train the model
        history = self.model.fit(
            train_data,
            validation_data=validation_data,
            epochs=epochs,
            batch_size=batch_size
        )
        
        return history
    
    def predict(self, image):
        """
        Predict the disease class for an input image
        
        Args:
            image (numpy.ndarray): Preprocessed input image
            
        Returns:
            tuple: (predicted_class_name, confidence_score)
        """
        # Ensure image has batch dimension
        if len(image.shape) == 3:
            image = np.expand_dims(image, axis=0)
        
        # Make prediction
        predictions = self.model.predict(image)
        
        # Get the predicted class index and confidence
        predicted_class_idx = np.argmax(predictions[0])
        confidence = float(predictions[0][predicted_class_idx])
        
        # Get the class name
        predicted_class = self.class_names.get(predicted_class_idx, f"Unknown Class {predicted_class_idx}")
        
        return predicted_class, confidence
    
    def save(self, model_path):
        """
        Save the model to a file
        
        Args:
            model_path (str): Path where to save the model
        """
        self.model.save(model_path)
        print(f"Model saved to {model_path}")
    
    def evaluate(self, test_data):
        """
        Evaluate the model on test data
        
        Args:
            test_data: Test data generator or dataset
            
        Returns:
            dict: Evaluation metrics
        """
        # Evaluate the model
        results = self.model.evaluate(test_data)
        
        # Create metrics dictionary
        metrics = {
            'loss': results[0],
            'accuracy': results[1]
        }
        
        return metrics
