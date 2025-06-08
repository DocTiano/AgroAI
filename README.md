# AgroAI - Advanced Crop Disease Detection System

AgroAI is a sophisticated web application that leverages deep learning and computer vision to detect crop diseases from images. The system helps farmers identify plant diseases early to prevent crop loss and optimize agricultural yields through accurate AI-powered diagnostics.

## Project Structure

- `app.py`: Main application file
- `app/`: Core application code
  - `models/`: AI model definitions and TensorFlow implementations
  - `templates/`: HTML templates
  - `static/`: CSS, JS, and images
  - `utils/`: Database and image processing utilities
- `instance/`: Contains the database
- `model_training/`: ML model training and evaluation scripts
- `dataset/`: Plant disease image dataset for training/testing
- `trained_models_*`: Trained model artifacts and performance metrics

## Machine Learning Architecture

The system uses a state-of-the-art convolutional neural network architecture:

- **Base Architecture**: MobileNetV2/EfficientNet (optimized for deployment)
- **Training Method**: Transfer learning with fine-tuning
- **Input**: 224x224 RGB images of plant leaves
- **Output**: 55 disease classes across multiple crop types
- **Performance**: >92% accuracy on validation data
- **Deployment**: Optimized TensorFlow Lite model for efficient inference

The model is trained on a comprehensive dataset containing over 60,000 images of plant diseases, with data augmentation techniques to improve generalization performance. The training process includes:

1. Feature extraction phase with frozen base layers (10 epochs)
2. Fine-tuning phase with unfrozen deeper layers (5 epochs)
3. Model evaluation and performance metrics tracking
4. Conversion to optimized TensorFlow Lite format

## Database Management

This project uses a fixed database structure without migrations. The database is automatically initialized when the application starts.

### Database Management Commands

The `manage_db.py` script provides commands to manage the database:

- **Backup**: Create a backup of the current database
  ```
  python manage_db.py backup
  ```

- **Restore**: Restore the database from a backup
  ```
  python manage_db.py restore
  ```

- **Verify**: Check if the database structure is correct
  ```
  python manage_db.py verify
  ```

- **Reset**: Reset the database to its initial state
  ```
  python manage_db.py reset
  ```

## Installation

1. Clone the repository
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Run the application:
   ```
   python app.py
   ```

## Model Training

To train a custom model on the dataset:

```
cd model_training
python train_custom_model.py
```

This will:
1. Load the plant disease dataset
2. Initialize a transfer learning model with MobileNetV2 architecture
3. Train the model with proper validation splits
4. Save the trained model and performance metrics
5. Convert the model to TensorFlow Lite format for deployment

## Features

- AI-powered image analysis and disease detection
- Disease identification with confidence scores
- Treatment and prevention recommendations
- Historical scan tracking and analytics
- User location-based features
- Community forum for farmers
- Multi-crop disease coverage (55+ disease classes)
- Mobile-optimized interface

## Technical Stack

- **Frontend**: HTML/CSS/JavaScript with Bootstrap
- **Backend**: Python/Flask
- **Database**: SQLite with SQLAlchemy ORM
- **AI/ML**: TensorFlow with transfer learning
- **Computer Vision**: OpenCV and PIL for preprocessing
- **Model Optimization**: TensorFlow Lite for efficient inference "# AgroAI" 
