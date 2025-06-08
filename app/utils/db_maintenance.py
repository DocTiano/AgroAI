"""
Database maintenance utilities for AgroAI
This module provides functions for database backup, restore, and verification
"""
import os
import shutil
import sqlite3
from datetime import datetime

def backup_database(app):
    """
    Create a backup of the current database
    
    Args:
        app: Flask application instance
    
    Returns:
        str: Path to the backup file
    """
    with app.app_context():
        # Get database path from app config
        db_uri = app.config['SQLALCHEMY_DATABASE_URI']
        if not db_uri.startswith('sqlite:///'):
            raise ValueError("Only SQLite databases are supported for backup")
        
        # Extract the path from the URI
        db_path = db_uri.replace('sqlite:///', '')
        
        # Create backups directory if it doesn't exist
        backup_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 
                                 'database', 'backups')
        os.makedirs(backup_dir, exist_ok=True)
        
        # Generate backup filename with timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        backup_path = os.path.join(backup_dir, f'agroai_backup_{timestamp}.db')
        
        # Copy the database file
        shutil.copy2(db_path, backup_path)
        
        print(f"Database backed up to: {backup_path}")
        return backup_path

def restore_database(app, backup_path=None):
    """
    Restore the database from a backup
    
    Args:
        app: Flask application instance
        backup_path (str, optional): Path to the backup file to restore.
            If None, the most recent backup will be used.
    
    Returns:
        bool: True if successful, False otherwise
    """
    with app.app_context():
        # Get database path from app config
        db_uri = app.config['SQLALCHEMY_DATABASE_URI']
        if not db_uri.startswith('sqlite:///'):
            raise ValueError("Only SQLite databases are supported for restore")
        
        # Extract the path from the URI
        db_path = db_uri.replace('sqlite:///', '')
        
        # If no backup path provided, find the most recent backup
        if not backup_path:
            backup_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 
                                     'database', 'backups')
            
            if not os.path.exists(backup_dir):
                print("No backups directory found")
                return False
            
            backups = [os.path.join(backup_dir, f) for f in os.listdir(backup_dir) 
                      if f.startswith('agroai_backup_') and f.endswith('.db')]
            
            if not backups:
                print("No backup files found")
                return False
            
            # Sort by modification time (most recent first)
            backups.sort(key=lambda x: os.path.getmtime(x), reverse=True)
            backup_path = backups[0]
        
        # Check if backup file exists
        if not os.path.exists(backup_path):
            print(f"Backup file not found: {backup_path}")
            return False
        
        try:
            # Stop the app from accessing the database
            from app.utils.database import db
            db.session.remove()
            db.engine.dispose()
            
            # Copy the backup over the current database
            shutil.copy2(backup_path, db_path)
            
            print(f"Database restored from: {backup_path}")
            return True
        except Exception as e:
            print(f"Error restoring database: {e}")
            return False

def verify_database_structure(app):
    """
    Verify the database structure is correct
    
    Args:
        app: Flask application instance
    
    Returns:
        bool: True if structure is correct, False otherwise
    """
    with app.app_context():
        # Get database path from app config
        db_uri = app.config['SQLALCHEMY_DATABASE_URI']
        if not db_uri.startswith('sqlite:///'):
            raise ValueError("Only SQLite databases are supported for verification")
        
        # Extract the path from the URI
        db_path = db_uri.replace('sqlite:///', '')
        
        # Connect to the database
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        try:
            # Check if required tables exist
            required_tables = ['user', 'disease', 'prediction', 'post']
            
            # Get list of tables
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
            tables = [row[0] for row in cursor.fetchall()]
            
            # Check if all required tables exist
            missing_tables = [table for table in required_tables if table not in tables]
            if missing_tables:
                print(f"Missing tables: {', '.join(missing_tables)}")
                return False
            
            # Check if prediction table has crop_type column
            cursor.execute("PRAGMA table_info(prediction)")
            columns = [column[1] for column in cursor.fetchall()]
            
            if 'crop_type' not in columns:
                print("Missing 'crop_type' column in prediction table")
                return False
            
            print("Database structure verification passed")
            return True
        except Exception as e:
            print(f"Error verifying database structure: {e}")
            return False
        finally:
            conn.close()

def create_backup_directory():
    """
    Create the backup directory if it doesn't exist
    
    Returns:
        str: Path to the backup directory
    """
    backup_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 
                             'database', 'backups')
    os.makedirs(backup_dir, exist_ok=True)
    return backup_dir 