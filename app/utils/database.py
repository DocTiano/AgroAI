"""
Database utilities for AgroAI
This module provides database models and initialization functions
"""
from flask_sqlalchemy import SQLAlchemy
from flask_login import UserMixin
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime
import os

# Initialize SQLAlchemy
db = SQLAlchemy()

def init_db(app):
    """
    Initialize the database with the Flask app
    
    Args:
        app: Flask application instance
    """
    db.init_app(app)
    
    with app.app_context():
        # Create tables if they don't exist
        db.create_all()
        
        # Seed initial data
        seed_diseases()

# Database Models
class User(db.Model, UserMixin):
    """User model for authentication and tracking predictions"""
    id = db.Column(db.Integer, primary_key=True)
    first_name = db.Column(db.String(50), nullable=True)
    last_name = db.Column(db.String(50), nullable=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password = db.Column(db.String(128), nullable=False)  # Plain text password
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    is_active = db.Column(db.Boolean, default=True)
    
    # Profile photo
    photo_path = db.Column(db.String(255), nullable=True)
    
    # Location fields for weather data
    province = db.Column(db.String(50))  # PSGC code for province
    municipality = db.Column(db.String(50))  # PSGC code for municipality
    barangay = db.Column(db.String(50))  # PSGC code for barangay
    latitude = db.Column(db.Float)
    longitude = db.Column(db.Float, nullable=True)
    
    # Relationships
    predictions = db.relationship('Prediction', backref='user', lazy=True)
    posts = db.relationship('Post', backref='author', lazy=True)
    
    def set_password(self, password):
        """Set the password for the user (plain text)"""
        self.password = password
    
    def check_password(self, password):
        """Check if the provided password matches"""
        return self.password == password
    
    def __repr__(self):
        return f'<User {self.username}>'

class Disease(db.Model):
    """Disease model for storing disease information"""
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    scientific_name = db.Column(db.String(100))
    description = db.Column(db.Text)
    symptoms = db.Column(db.Text)
    treatment = db.Column(db.Text)
    prevention = db.Column(db.Text)
    
    # Relationships
    predictions = db.relationship('Prediction', backref='disease', lazy=True)
    
    def __repr__(self):
        return f'<Disease {self.name}>'

class Post(db.Model):
    """Post model for community forum"""
    id = db.Column(db.Integer, primary_key=True)
    title = db.Column(db.String(200), nullable=False)
    content = db.Column(db.Text, nullable=False)
    category = db.Column(db.String(50), nullable=False, default='General')
    tags = db.Column(db.String(200), nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    views = db.Column(db.Integer, default=0)
    comments_count = db.Column(db.Integer, default=0)
    
    def __repr__(self):
        return f'<Post {self.id} by User {self.user_id}> {self.title[:20]}...'
        
    def get_tags_list(self):
        """Return tags as a list"""
        if not self.tags:
            return []
        return [tag.strip() for tag in self.tags.split(',')]

class Prediction(db.Model):
    """Prediction model for storing user prediction history"""
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    disease_id = db.Column(db.Integer, db.ForeignKey('disease.id'), nullable=False)
    image_path = db.Column(db.String(255), nullable=False)
    confidence = db.Column(db.Float, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    crop_type = db.Column(db.String(50), nullable=True, default='Tomato')  # Default to Tomato as per user's request
    
    def __repr__(self):
        return f'<Prediction {self.id} by User {self.user_id}>'

def seed_diseases():
    """
    Seed the database with common plant diseases
    This function should be called once after initializing the database
    """
    # Check if diseases already exist
    if Disease.query.count() > 0:
        return
    
    # List of common plant diseases with information
    diseases = [
        {
            'name': 'Tomato Late Blight',
            'scientific_name': 'Phytophthora infestans',
            'description': 'Late blight is a devastating disease of tomato and potato that caused the Irish potato famine in the 1840s.',
            'symptoms': 'Dark, water-soaked lesions on leaves, stems, and fruits. White, fuzzy growth on the underside of leaves in humid conditions.',
            'treatment': 'Remove and destroy infected plants. Apply fungicides containing copper or chlorothalonil.',
            'prevention': 'Plant resistant varieties. Avoid overhead irrigation. Ensure good air circulation. Apply preventive fungicides during humid weather.'
        },
        {
            'name': 'Apple Scab',
            'scientific_name': 'Venturia inaequalis',
            'description': 'Apple scab is one of the most common diseases of apple trees, affecting both leaves and fruit.',
            'symptoms': 'Olive-green to brown spots on leaves and fruit. Infected leaves may drop prematurely. Scabby, deformed fruit.',
            'treatment': 'Apply fungicides containing captan or myclobutanil. Remove and destroy fallen leaves.',
            'prevention': 'Plant resistant varieties. Rake and destroy fallen leaves. Prune to improve air circulation. Apply preventive fungicides in spring.'
        },
        {
            'name': 'Corn Northern Leaf Blight',
            'scientific_name': 'Exserohilum turcicum',
            'description': 'Northern leaf blight is a fungal disease that affects corn, reducing yield and quality.',
            'symptoms': 'Long, elliptical, grayish-green to tan lesions on leaves. Lesions may coalesce, causing entire leaves to die.',
            'treatment': 'Apply fungicides containing azoxystrobin or propiconazole. Remove and destroy infected crop debris.',
            'prevention': 'Plant resistant hybrids. Rotate crops. Plow under crop debris. Apply preventive fungicides at early signs of disease.'
        },
        {
            'name': 'Healthy',
            'scientific_name': 'No disease',
            'description': 'No detailed information available for this disease.',
            'symptoms': 'Please consult with an agricultural expert for more information.',
            'treatment': 'Please consult with an agricultural expert for treatment options.',
            'prevention': 'Please consult with an agricultural expert for prevention strategies.'
        },
        {
            'name': 'Tomato Early Blight',
            'scientific_name': 'Alternaria solani',
            'description': 'Early blight is a common fungal disease that affects tomato plants, particularly in warm, humid conditions.',
            'symptoms': 'Dark, concentric rings on lower leaves forming a target-like pattern. Leaves may yellow and drop prematurely.',
            'treatment': 'Remove infected leaves. Apply fungicides containing chlorothalonil or copper. Ensure proper plant spacing.',
            'prevention': 'Mulch around plants. Avoid wetting foliage when watering. Practice crop rotation. Remove plant debris after harvest.'
        },
        {
            'name': 'Rice Blast',
            'scientific_name': 'Magnaporthe oryzae',
            'description': 'Rice blast is one of the most destructive diseases of rice worldwide, affecting all above-ground parts of the plant.',
            'symptoms': 'Diamond-shaped lesions on leaves. White to gray-green lesions with dark borders. Infected panicles may break.',
            'treatment': 'Apply fungicides containing azoxystrobin or tricyclazole. Remove and destroy infected plants.',
            'prevention': 'Plant resistant varieties. Maintain balanced fertilization. Avoid excessive nitrogen. Improve field drainage.'
        },
        {
            'name': 'Potato Late Blight',
            'scientific_name': 'Phytophthora infestans',
            'description': 'Late blight is a devastating disease of potato and tomato that caused the Irish potato famine in the 1840s.',
            'symptoms': 'Water-soaked black/brown lesions on leaves and stems. White fungal growth on leaf undersides in humid conditions.',
            'treatment': 'Remove and destroy infected plants. Apply fungicides containing copper or chlorothalonil.',
            'prevention': 'Plant resistant varieties. Use certified disease-free seed potatoes. Ensure good air circulation.'
        }
    ]
    
    # Add diseases to database
    for disease_data in diseases:
        disease = Disease(**disease_data)
        db.session.add(disease)
    
    # Commit changes
    db.session.commit()
    print(f"Seeded database with {len(diseases)} diseases")

def ensure_database_structure():
    """
    Ensure the database has the correct structure
    This function handles any structural changes that would normally be done via migrations
    """
    # This would contain any structural changes needed
    # Currently handled in app.py during initialization
    pass
