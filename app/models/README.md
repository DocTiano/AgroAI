# Plant Disease Detection Models

This directory contains the machine learning models and supporting code for the AgroAI plant disease detection system.

## Core Components

- **plant_disease_detector.py**: The main model wrapper that provides the classification interface
- **disease_model.py**: Base model class for all detection models
- **train_transfer_learning.py**: Transfer learning pipeline for custom model training
- **train_model.py**: General-purpose model training utilities

## Model Architecture

The system uses transfer learning with pre-trained CNN architectures:

1. **Base Architecture**: MobileNetV2 or EfficientNetB0 (configurable)
2. **Custom Head**: Added classification layers
   - Global Average Pooling
   - Dense layers (1024 -> 512 -> num_classes)
   - Dropout for regularization (0.5, 0.3)
   - Softmax activation

## Training Process

The models are trained using a two-phase approach:

1. **Feature Extraction**: Train only the custom head with frozen base layers
2. **Fine-Tuning**: Train selected deeper layers with lower learning rate

Training parameters:
- Input size: 224x224 RGB images
- Batch size: 16-32 (auto-adjusts based on available memory)
- Learning rate: 0.001 (initial), 0.00001 (fine-tuning)
- Epochs: 10 (initial) + 5 (fine-tuning)
- Data augmentation: rotation, zoom, shift, flip

## Model Deployment

Models are converted to TensorFlow Lite for efficient deployment:
- Reduced model size (~4-10MB)
- Optimized inference speed
- Compatible with mobile and edge devices

## Disease Coverage

The model can detect 55+ plant diseases across multiple crops:
- Tomato (10 diseases)
- Apple (4 diseases)
- Potato (3 diseases)
- Corn (4 diseases)
- Grape (4 diseases)
- Mango (7 diseases)
- Banana (3 diseases)
- Wheat (12 diseases)
- Others (8+ diseases)

## Performance Metrics

On the validation dataset:
- Accuracy: 92-96%
- Precision: 91-95%
- Recall: 90-94%
- F1 Score: 91-94%

## Backup System

The system includes a robust fallback mechanism that ensures service continuity even if the main detection model fails to load. The backup system:

1. Uses a simplified architecture for reliability
2. Provides deterministic predictions based on plant type
3. Maintains base functionality with reduced capabilities
4. Automatically logs issues for maintenance

## Custom Model Training

To train a custom model with your own dataset:

```python
from app.models.train_transfer_learning import TransferLearningTrainer

trainer = TransferLearningTrainer(
    dataset_path="path/to/dataset",
    batch_size=32,
    img_size=224
)

# Train model
trainer.train_model(epochs=10, fine_tune=True, fine_tune_epochs=5)

# Save model
trainer.save_model("path/to/save/model")
```

## Integration

The models integrate with the main application through a standardized API:
- `predict(image_path, plant_type=None)`: Returns disease classification and confidence
- `detect_disease_areas(image_path)`: Returns visualization of affected areas
- `get_crop_type_from_disease(disease_name)`: Extracts crop type from disease name 