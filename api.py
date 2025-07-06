# ===== Production Web API for Crop Disease Detection =====
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
import io
import base64
import json
import logging
import os
from datetime import datetime
import uuid
from werkzeug.utils import secure_filename
from functools import wraps
import sqlite3
from contextlib import contextmanager
import threading

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class CropDiseaseAPI:
    def __init__(self, models_dir='models', upload_dir='uploads', db_path='predictions.db'):
        self.app = Flask(__name__)
        self.app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
        
        # Enable CORS for all domains (configure for production)
        CORS(self.app, origins=["*"])
        
        # Rate limiting
        self.limiter = Limiter(
            app=self.app,
            key_func=get_remote_address,
            default_limits=["200 per day", "50 per hour"]
        )
        
        # Directories
        self.models_dir = models_dir
        self.upload_dir = upload_dir
        self.db_path = db_path
        
        # Create directories
        os.makedirs(models_dir, exist_ok=True)
        os.makedirs(upload_dir, exist_ok=True)
        
        # Initialize database
        self.init_database()
        
        # Load models
        self.models = {}
        self.class_names = {}
        self.load_models()
        
        # Thread lock for model inference
        self.model_lock = threading.Lock()
        
        # Setup routes
        self.setup_routes()
        
        logger.info("CropDiseaseAPI initialized successfully")
    
    def init_database(self):
        """Initialize SQLite database for logging predictions"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                CREATE TABLE IF NOT EXISTS predictions (
                    id TEXT PRIMARY KEY,
                    timestamp TEXT,
                    crop TEXT,
                    model_used TEXT,
                    predicted_disease TEXT,
                    confidence REAL,
                    top3_predictions TEXT,
                    image_path TEXT,
                    user_agent TEXT,
                    ip_address TEXT,
                    processing_time_ms REAL
                )
            ''')
            
            conn.execute('''
                CREATE TABLE IF NOT EXISTS feedback (
                    id TEXT PRIMARY KEY,
                    prediction_id TEXT,
                    actual_disease TEXT,
                    user_feedback TEXT,
                    timestamp TEXT,
                    FOREIGN KEY (prediction_id) REFERENCES predictions (id)
                )
            ''')
            conn.commit()
    
    @contextmanager
    def get_db_connection(self):
        """Context manager for database connections"""
        conn = sqlite3.connect(self.db_path)
        try:
            yield conn
        finally:
            conn.close()
    
    def load_models(self):
        """Load all available trained models"""
        logger.info("Loading trained models...")
        
        # Model configuration - update paths after training
        model_configs = {
            'cashew': {
                'model_path': os.path.join(self.models_dir, 'cashew_enhanced_mobilenet.h5'),
                'classes': ['anthracnose', 'gumosis', 'healthy', 'leaf miner', 'red rust']
            },

            'cassava': {
                'model_path': os.path.join(self.models_dir, 'cassava_enhanced_mobilenet.h5'),
                'classes': ['bacterial blight', 'brown spot', 'green mite', 'healthy', 'mosaic']
            },
            'maize': {
                'model_path': os.path.join(self.models_dir, 'maize_enhanced_mobilenet.h5'),
                'classes': ['fall armyworm', 'grasshoper', 'healthy', 'leaf beetle', 'leaf blight', 'leaf spot', 'streak virus']
            },
            'tomato': {
                'model_path': os.path.join(self.models_dir, 'tomato_enhanced_mobilenet.h5'),
                'classes': ['healthy', 'leaf blight', 'leaf curl', 'septoria leaf spot', 'verticulium wilt']
            }
        }
        
        for crop, config in model_configs.items():
            try:
                if os.path.exists(config['model_path']):
                    # Load model
                    model = tf.keras.models.load_model(config['model_path'])
                    self.models[crop] = model
                    self.class_names[crop] = config['classes']
                    logger.info(f"Loaded model for {crop}: {len(config['classes'])} classes")
                else:
                    logger.warning(f"Model not found for {crop}: {config['model_path']}")
                    # Create dummy model info for testing
                    self.models[crop] = None
                    self.class_names[crop] = config['classes']
            
            except Exception as e:
                logger.error(f"Error loading model for {crop}: {str(e)}")
        
        logger.info(f"Loaded {len([m for m in self.models.values() if m is not None])} models successfully")
    
    def preprocess_image(self, image_data, target_size=(224, 224)):
        """Preprocess image for model inference"""
        try:
            # Handle different input types
            if isinstance(image_data, str):
                # Base64 encoded image
                image_bytes = base64.b64decode(image_data)
                image = Image.open(io.BytesIO(image_bytes))
            elif hasattr(image_data, 'read'):
                # File-like object
                image = Image.open(image_data)
            else:
                # PIL Image or numpy array
                image = image_data
            
            # Convert to RGB if necessary
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Resize image
            image = image.resize(target_size, Image.LANCZOS)
            
            # Convert to numpy array and normalize
            image_array = np.array(image, dtype=np.float32) / 255.0
            
            # Add batch dimension
            image_array = np.expand_dims(image_array, axis=0)
            
            return image_array
            
        except Exception as e:
            logger.error(f"Error preprocessing image: {str(e)}")
            raise ValueError(f"Invalid image data: {str(e)}")
    
    def predict_disease(self, crop, image_data, model_preference='auto'):
        """Predict disease for given crop and image"""
        start_time = datetime.now()
        
        try:
            # Validate crop
            if crop.lower() not in self.models:
                raise ValueError(f"Unsupported crop: {crop}. Available: {list(self.models.keys())}")
            
            crop_lower = crop.lower()
            
            # Check if model is available
            if self.models[crop_lower] is None:
                # Return dummy prediction for testing
                return {
                    'crop': crop,
                    'predicted_disease': 'healthy',
                    'confidence': 0.85,
                    'top_predictions': [
                        {'disease': 'healthy', 'confidence': 0.85},
                        {'disease': 'leaf blight', 'confidence': 0.10},
                        {'disease': 'mosaic', 'confidence': 0.05}
                    ],
                    'model_used': 'dummy_model',
                    'processing_time_ms': 50,
                    'status': 'success'
                }
            
            # Preprocess image
            processed_image = self.preprocess_image(image_data)
            
            # Make prediction with thread safety
            with self.model_lock:
                predictions = self.models[crop_lower].predict(processed_image, verbose=0)
            
            # Process predictions
            prediction_probs = predictions[0]
            class_names = self.class_names[crop_lower]
            
            # Get top 3 predictions
            top_indices = np.argsort(prediction_probs)[-3:][::-1]
            
            top_predictions = []
            for idx in top_indices:
                disease = class_names[idx]
                confidence = float(prediction_probs[idx])
                top_predictions.append({
                    'disease': disease,
                    'confidence': confidence
                })
            
            # Calculate processing time
            processing_time = (datetime.now() - start_time).total_seconds() * 1000
            
            result = {
                'crop': crop,
                'predicted_disease': top_predictions[0]['disease'],
                'confidence': top_predictions[0]['confidence'],
                'top_predictions': top_predictions,
                'model_used': f'enhanced_mobilenet_{crop_lower}',
                'processing_time_ms': processing_time,
                'status': 'success'
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error in prediction: {str(e)}")
            processing_time = (datetime.now() - start_time).total_seconds() * 1000
            
            return {
                'crop': crop,
                'error': str(e),
                'processing_time_ms': processing_time,
                'status': 'error'
            }
    
    def log_prediction(self, prediction_result, image_path=None, user_agent=None, ip_address=None):
        """Log prediction to database"""
        try:
            prediction_id = str(uuid.uuid4())
            
            with self.get_db_connection() as conn:
                conn.execute('''
                    INSERT INTO predictions 
                    (id, timestamp, crop, model_used, predicted_disease, confidence, 
                     top3_predictions, image_path, user_agent, ip_address, processing_time_ms)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    prediction_id,
                    datetime.now().isoformat(),
                    prediction_result.get('crop'),
                    prediction_result.get('model_used'),
                    prediction_result.get('predicted_disease'),
                    prediction_result.get('confidence'),
                    json.dumps(prediction_result.get('top_predictions', [])),
                    image_path,
                    user_agent,
                    ip_address,
                    prediction_result.get('processing_time_ms')
                ))
                conn.commit()
            
            return prediction_id
            
        except Exception as e:
            logger.error(f"Error logging prediction: {str(e)}")
            return None
    
    def setup_routes(self):
        """Setup all API routes"""
        
        @self.app.route('/', methods=['GET'])
        def home():
            """API documentation"""
            return jsonify({
                'service': 'Crop Disease Detection API',
                'version': '1.0.0',
                'status': 'operational',
                'available_crops': list(self.models.keys()),
                'endpoints': {
                    'POST /predict': 'Predict disease from image',
                    'POST /predict/batch': 'Batch prediction for multiple images',
                    'GET /crops': 'Get available crops and classes',
                    'GET /health': 'Health check',
                    'POST /feedback': 'Submit prediction feedback',
                    'GET /stats': 'API usage statistics'
                },
                'documentation': 'https://your-domain.com/docs'
            })
        
        @self.app.route('/health', methods=['GET'])
        def health_check():
            """Health check endpoint"""
            models_loaded = len([m for m in self.models.values() if m is not None])
            
            return jsonify({
                'status': 'healthy',
                'timestamp': datetime.now().isoformat(),
                'models_loaded': models_loaded,
                'total_models': len(self.models),
                'gpu_available': len(tf.config.list_physical_devices('GPU')) > 0,
                'tensorflow_version': tf.__version__
            })
        
        @self.app.route('/crops', methods=['GET'])
        def get_crops():
            """Get available crops and their disease classes"""
            crops_info = {}
            for crop, classes in self.class_names.items():
                crops_info[crop] = {
                    'classes': classes,
                    'model_loaded': self.models[crop] is not None,
                    'num_classes': len(classes)
                }
            
            return jsonify({
                'available_crops': crops_info,
                'total_crops': len(crops_info)
            })
        
        @self.app.route('/predict', methods=['POST'])
        @self.limiter.limit("30 per minute")
        def predict():
            """Main prediction endpoint"""
            try:
                # Get request data
                if request.content_type.startswith('multipart/form-data'):
                    # File upload
                    if 'image' not in request.files:
                        return jsonify({'error': 'No image file provided'}), 400
                    
                    image_file = request.files['image']
                    if image_file.filename == '':
                        return jsonify({'error': 'No image file selected'}), 400
                    
                    crop = request.form.get('crop', '').lower()
                    
                    # Save uploaded file
                    filename = secure_filename(f"{uuid.uuid4()}_{image_file.filename}")
                    image_path = os.path.join(self.upload_dir, filename)
                    image_file.save(image_path)
                    
                    # Use file for prediction
                    image_data = image_file
                    
                elif request.content_type == 'application/json':
                    # JSON with base64 image
                    data = request.get_json()
                    
                    if not data or 'image' not in data or 'crop' not in data:
                        return jsonify({'error': 'Missing required fields: image, crop'}), 400
                    
                    crop = data['crop'].lower()
                    image_data = data['image']
                    image_path = None
                    
                else:
                    return jsonify({'error': 'Unsupported content type'}), 400
                
                # Validate crop
                if crop not in self.models:
                    return jsonify({
                        'error': f'Unsupported crop: {crop}',
                        'available_crops': list(self.models.keys())
                    }), 400
                
                # Make prediction
                result = self.predict_disease(crop, image_data)
                
                # Log prediction
                prediction_id = self.log_prediction(
                    result,
                    image_path=image_path,
                    user_agent=request.headers.get('User-Agent'),
                    ip_address=request.remote_addr
                )
                
                # Add prediction ID to result
                if prediction_id:
                    result['prediction_id'] = prediction_id
                
                # Return result
                if result['status'] == 'success':
                    return jsonify(result), 200
                else:
                    return jsonify(result), 500
                
            except Exception as e:
                logger.error(f"Error in predict endpoint: {str(e)}")
                return jsonify({
                    'error': 'Internal server error',
                    'message': str(e),
                    'status': 'error'
                }), 500
        
        @self.app.route('/predict/batch', methods=['POST'])
        @self.limiter.limit("5 per minute")
        def batch_predict():
            """Batch prediction endpoint"""
            try:
                data = request.get_json()
                
                if not data or 'predictions' not in data:
                    return jsonify({'error': 'Missing predictions array'}), 400
                
                predictions_input = data['predictions']
                if len(predictions_input) > 10:  # Limit batch size
                    return jsonify({'error': 'Maximum 10 predictions per batch'}), 400
                
                results = []
                
                for i, pred_data in enumerate(predictions_input):
                    if 'image' not in pred_data or 'crop' not in pred_data:
                        results.append({
                            'index': i,
                            'error': 'Missing required fields: image, crop',
                            'status': 'error'
                        })
                        continue
                    
                    result = self.predict_disease(pred_data['crop'], pred_data['image'])
                    result['index'] = i
                    results.append(result)
                
                return jsonify({
                    'batch_results': results,
                    'total_predictions': len(results),
                    'successful_predictions': len([r for r in results if r.get('status') == 'success'])
                }), 200
                
            except Exception as e:
                logger.error(f"Error in batch predict: {str(e)}")
                return jsonify({'error': 'Internal server error'}), 500
        
        @self.app.route('/feedback', methods=['POST'])
        def submit_feedback():
            """Submit feedback for a prediction"""
            try:
                data = request.get_json()
                
                required_fields = ['prediction_id', 'actual_disease']
                if not data or not all(field in data for field in required_fields):
                    return jsonify({'error': f'Missing required fields: {required_fields}'}), 400
                
                feedback_id = str(uuid.uuid4())
                
                with self.get_db_connection() as conn:
                    conn.execute('''
                        INSERT INTO feedback (id, prediction_id, actual_disease, user_feedback, timestamp)
                        VALUES (?, ?, ?, ?, ?)
                    ''', (
                        feedback_id,
                        data['prediction_id'],
                        data['actual_disease'],
                        data.get('user_feedback', ''),
                        datetime.now().isoformat()
                    ))
                    conn.commit()
                
                return jsonify({
                    'feedback_id': feedback_id,
                    'status': 'success',
                    'message': 'Feedback submitted successfully'
                }), 200
                
            except Exception as e:
                logger.error(f"Error submitting feedback: {str(e)}")
                return jsonify({'error': 'Internal server error'}), 500
        
        @self.app.route('/stats', methods=['GET'])
        def get_stats():
            """Get API usage statistics"""
            try:
                with self.get_db_connection() as conn:
                    # Total predictions
                    total_predictions = conn.execute('SELECT COUNT(*) FROM predictions').fetchone()[0]
                    
                    # Predictions by crop
                    crop_stats = conn.execute('''
                        SELECT crop, COUNT(*) as count 
                        FROM predictions 
                        GROUP BY crop
                    ''').fetchall()
                    
                    # Recent predictions (last 24 hours)
                    recent_predictions = conn.execute('''
                        SELECT COUNT(*) FROM predictions 
                        WHERE datetime(timestamp) > datetime('now', '-1 day')
                    ''').fetchone()[0]
                    
                    # Average processing time
                    avg_processing_time = conn.execute('''
                        SELECT AVG(processing_time_ms) FROM predictions 
                        WHERE processing_time_ms IS NOT NULL
                    ''').fetchone()[0] or 0
                
                return jsonify({
                    'total_predictions': total_predictions,
                    'recent_predictions_24h': recent_predictions,
                    'average_processing_time_ms': round(avg_processing_time, 2),
                    'predictions_by_crop': dict(crop_stats),
                    'models_available': len([m for m in self.models.values() if m is not None]),
                    'timestamp': datetime.now().isoformat()
                }), 200
                
            except Exception as e:
                logger.error(f"Error getting stats: {str(e)}")
                return jsonify({'error': 'Internal server error'}), 500
        
        # Error handlers
        @self.app.errorhandler(413)
        def file_too_large(error):
            return jsonify({'error': 'File too large. Maximum size is 16MB.'}), 413
        
        @self.app.errorhandler(429)
        def rate_limit_exceeded(error):
            return jsonify({'error': 'Rate limit exceeded. Please try again later.'}), 429
        
        @self.app.errorhandler(500)
        def internal_error(error):
            return jsonify({'error': 'Internal server error'}), 500
    
    def run(self, host='0.0.0.0', port=5000, debug=False):
        """Run the Flask application"""
        logger.info(f"Starting Crop Disease Detection API on {host}:{port}")
        self.app.run(host=host, port=port, debug=debug, threaded=True)

# Model saving utility for after training
def save_model_for_api(model, crop_name, model_type='enhanced_mobilenet', models_dir='models'):
    """Save trained model for API deployment"""
    os.makedirs(models_dir, exist_ok=True)
    
    model_filename = f"{crop_name}_{model_type}.h5"
    model_path = os.path.join(models_dir, model_filename)
    
    # Save model
    model.save(model_path)
    
    print(f"Model saved for API: {model_path}")
    return model_path

# Production deployment script
if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Crop Disease Detection API')
    parser.add_argument('--host', default='0.0.0.0', help='Host to run on')
    parser.add_argument('--port', type=int, default=5000, help='Port to run on')
    parser.add_argument('--debug', action='store_true', help='Run in debug mode')
    parser.add_argument('--models-dir', default='models', help='Directory containing models')
    
    args = parser.parse_args()
    
    # Initialize and run API
    api = CropDiseaseAPI(models_dir=args.models_dir)
    api.run(host=args.host, port=args.port, debug=args.debug)