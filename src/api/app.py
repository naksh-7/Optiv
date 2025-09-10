# Flask REST API for SecureDoc AI
from flask import Flask, request, jsonify, send_file
from flask_restx import Api, Resource, fields, Namespace
from flask_cors import CORS
import cv2
import numpy as np
from PIL import Image
import io
import base64
import json
import logging
from typing import Dict, Any, List
import traceback
from pathlib import Path
import uuid
import os
from werkzeug.utils import secure_filename

# Import core modules
from ..core.document_processor import DocumentProcessor, DocumentProcessingResult
from ..utils.config import get_api_config

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# API configuration
api_config = get_api_config()
api = Api(
    app,
    version='1.0',
    title='SecureDoc AI API',
    description='Advanced Document PII Detection & Logo Removal API',
    doc='/docs/'
)

# Create namespace
ns = Namespace('api/v1', description='SecureDoc AI operations')
api.add_namespace(ns)

# Global processor instance
processor = None

# Request/Response models for API documentation
upload_model = api.model('DocumentUpload', {
    'image': fields.String(required=True, description='Base64 encoded image'),
    'filename': fields.String(required=False, description='Original filename'),
    'options': fields.Raw(required=False, description='Processing options')
})

processing_options_model = api.model('ProcessingOptions', {
    'ocr_confidence': fields.Float(description='OCR confidence threshold (0-1)', default=0.7),
    'pii_confidence': fields.Float(description='PII confidence threshold (0-1)', default=0.8),
    'logo_confidence': fields.Float(description='Logo confidence threshold (0-1)', default=0.6),
    'pii_engines': fields.List(fields.String, description='PII detection engines', default=['presidio', 'spacy']),
    'logo_removal_method': fields.String(description='Logo removal method', default='inpaint'),
    'preserve_format': fields.Boolean(description='Preserve text format', default=True),
    'realistic_replacements': fields.Boolean(description='Use realistic replacements', default=True)
})

entity_model = api.model('PIIEntity', {
    'text': fields.String(description='Entity text'),
    'label': fields.String(description='Entity type'),
    'start': fields.Integer(description='Start position'),
    'end': fields.Integer(description='End position'),
    'confidence': fields.Float(description='Confidence score'),
    'engine': fields.String(description='Detection engine'),
    'is_sensitive': fields.Boolean(description='Is sensitive entity')
})

logo_model = api.model('LogoDetection', {
    'class_name': fields.String(description='Logo class name'),
    'confidence': fields.Float(description='Detection confidence'),
    'bbox': fields.List(fields.Integer, description='Bounding box [x1, y1, x2, y2]'),
    'area': fields.Float(description='Detection area')
})

metrics_model = api.model('ProcessingMetrics', {
    'total_time': fields.Float(description='Total processing time'),
    'ocr_time': fields.Float(description='OCR processing time'),
    'pii_detection_time': fields.Float(description='PII detection time'),
    'logo_detection_time': fields.Float(description='Logo detection time'),
    'anonymization_time': fields.Float(description='Anonymization time'),
    'confidence_scores': fields.Raw(description='Confidence scores by engine'),
    'entities_found': fields.Integer(description='Number of entities found'),
    'logos_found': fields.Integer(description='Number of logos found')
})

response_model = api.model('ProcessingResponse', {
    'success': fields.Boolean(description='Processing success'),
    'document_id': fields.String(description='Document identifier'),
    'extracted_text': fields.String(description='Extracted text'),
    'anonymized_text': fields.String(description='Anonymized text'),
    'entities': fields.List(fields.Nested(entity_model), description='Detected PII entities'),
    'logos': fields.List(fields.Nested(logo_model), description='Detected logos'),
    'processing_metrics': fields.Nested(metrics_model, description='Processing metrics'),
    'anonymization_mapping': fields.Raw(description='Original to anonymized text mapping'),
    'processed_image_base64': fields.String(description='Base64 encoded processed image'),
    'error_message': fields.String(description='Error message if failed')
})

# Utility functions
def init_processor():
    """Initialize document processor"""
    global processor
    if processor is None:
        try:
            config = {
                "ocr": {
                    "primary_engine": "easyocr",
                    "fallback_engine": "tesseract",
                    "confidence_threshold": 0.7
                },
                "pii_detection": {
                    "engines": ["presidio", "spacy", "custom_patterns"],
                    "confidence_threshold": 0.8,
                    "enable_context_analysis": True
                },
                "logo_detection": {
                    "confidence_threshold": 0.6,
                    "removal_method": "inpaint"
                },
                "anonymization": {
                    "preserve_format": True,
                    "use_realistic_replacements": True
                }
            }
            processor = DocumentProcessor(config)
            logger.info("Document processor initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize processor: {e}")
            raise

def decode_base64_image(base64_string: str) -> np.ndarray:
    """Decode base64 string to numpy array image"""
    try:
        # Remove header if present
        if ',' in base64_string:
            base64_string = base64_string.split(',')[1]
        
        # Decode base64
        image_data = base64.b64decode(base64_string)
        
        # Convert to PIL Image
        pil_image = Image.open(io.BytesIO(image_data))
        
        # Convert to numpy array
        image_array = np.array(pil_image)
        
        # Convert RGB to BGR for OpenCV
        if len(image_array.shape) == 3 and image_array.shape[2] == 3:
            image_array = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
        
        return image_array
        
    except Exception as e:
        raise ValueError(f"Invalid base64 image data: {e}")

def encode_image_to_base64(image: np.ndarray) -> str:
    """Encode numpy array image to base64 string"""
    try:
        # Convert BGR to RGB
        if len(image.shape) == 3 and image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Convert to PIL Image
        pil_image = Image.fromarray(image)
        
        # Save to bytes
        buffer = io.BytesIO()
        pil_image.save(buffer, format='JPEG')
        buffer.seek(0)
        
        # Encode to base64
        image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        
        return image_base64
        
    except Exception as e:
        logger.error(f"Error encoding image to base64: {e}")
        return ""

def create_processing_config(options: Dict[str, Any]) -> Dict[str, Any]:
    """Create processing configuration from API options"""
    return {
        "ocr": {
            "confidence_threshold": options.get("ocr_confidence", 0.7)
        },
        "pii_detection": {
            "engines": options.get("pii_engines", ["presidio", "spacy", "custom_patterns"]),
            "confidence_threshold": options.get("pii_confidence", 0.8)
        },
        "logo_detection": {
            "confidence_threshold": options.get("logo_confidence", 0.6),
            "removal_method": options.get("logo_removal_method", "inpaint")
        },
        "anonymization": {
            "preserve_format": options.get("preserve_format", True),
            "use_realistic_replacements": options.get("realistic_replacements", True)
        }
    }

def format_processing_result(result: DocumentProcessingResult) -> Dict[str, Any]:
    """Format processing result for API response"""
    try:
        response = {
            "success": result.success,
            "document_id": result.document_id,
            "extracted_text": result.extracted_text,
            "anonymized_text": result.final_text,
            "entities": [
                {
                    "text": entity.text,
                    "label": entity.label,
                    "start": entity.start,
                    "end": entity.end,
                    "confidence": entity.confidence,
                    "engine": entity.engine,
                    "is_sensitive": entity.is_sensitive
                }
                for entity in result.detected_entities
            ],
            "logos": [
                {
                    "class_name": logo.class_name,
                    "confidence": logo.confidence,
                    "bbox": list(logo.bbox),
                    "area": logo.area
                }
                for logo in result.logo_detections
            ],
            "processing_metrics": {
                "total_time": result.processing_metrics.total_time,
                "ocr_time": result.processing_metrics.ocr_time,
                "pii_detection_time": result.processing_metrics.pii_detection_time,
                "logo_detection_time": result.processing_metrics.logo_detection_time,
                "anonymization_time": result.processing_metrics.anonymization_time,
                "confidence_scores": result.processing_metrics.confidence_scores,
                "entities_found": result.processing_metrics.entities_found,
                "logos_found": result.processing_metrics.logos_found
            },
            "anonymization_mapping": result.anonymization_result.mapping if result.anonymization_result else {},
            "processed_image_base64": "",
            "error_message": result.error_message
        }
        
        # Encode processed image if available
        if result.final_image is not None:
            response["processed_image_base64"] = encode_image_to_base64(result.final_image)
        
        return response
        
    except Exception as e:
        logger.error(f"Error formatting result: {e}")
        return {
            "success": False,
            "error_message": f"Error formatting result: {e}"
        }

# API Routes
@ns.route('/health')
class HealthCheck(Resource):
    def get(self):
        """Health check endpoint"""
        try:
            init_processor()
            return {
                "status": "healthy",
                "processor_initialized": processor is not None,
                "timestamp": pd.Timestamp.now().isoformat()
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "timestamp": pd.Timestamp.now().isoformat()
            }, 500

@ns.route('/process')
class ProcessDocument(Resource):
    @ns.expect(upload_model)
    @ns.marshal_with(response_model)
    def post(self):
        """Process a document for PII detection and logo removal"""
        try:
            # Initialize processor if needed
            init_processor()
            
            # Get request data
            data = request.get_json()
            
            if not data or 'image' not in data:
                return {"success": False, "error_message": "No image data provided"}, 400
            
            # Decode image
            try:
                image = decode_base64_image(data['image'])
            except ValueError as e:
                return {"success": False, "error_message": str(e)}, 400
            
            # Get processing options
            options = data.get('options', {})
            processing_config = create_processing_config(options)
            
            # Generate document ID
            document_id = data.get('filename', f"doc_{uuid.uuid4().hex[:8]}")
            
            # Process document
            result = processor.process_document(
                image,
                document_id=document_id,
                processing_options=processing_config
            )
            
            # Format response
            response = format_processing_result(result)
            
            return response
            
        except Exception as e:
            logger.error(f"Error processing document: {e}")
            logger.error(traceback.format_exc())
            return {
                "success": False,
                "error_message": f"Internal server error: {str(e)}"
            }, 500

@ns.route('/batch-process')
class BatchProcessDocuments(Resource):
    def post(self):
        """Process multiple documents in batch"""
        try:
            # Initialize processor if needed
            init_processor()
            
            # Get request data
            data = request.get_json()
            
            if not data or 'images' not in data:
                return {"success": False, "error_message": "No image data provided"}, 400
            
            images_data = data['images']
            if not isinstance(images_data, list):
                return {"success": False, "error_message": "Images must be a list"}, 400
            
            # Process each image
            results = []
            options = data.get('options', {})
            processing_config = create_processing_config(options)
            
            for i, image_data in enumerate(images_data):
                try:
                    # Decode image
                    image = decode_base64_image(image_data.get('image', ''))
                    document_id = image_data.get('filename', f"batch_doc_{i}")
                    
                    # Process document
                    result = processor.process_document(
                        image,
                        document_id=document_id,
                        processing_options=processing_config
                    )
                    
                    # Format response
                    formatted_result = format_processing_result(result)
                    results.append(formatted_result)
                    
                except Exception as e:
                    logger.error(f"Error processing document {i}: {e}")
                    results.append({
                        "success": False,
                        "document_id": f"batch_doc_{i}",
                        "error_message": str(e)
                    })
            
            # Generate batch statistics
            successful = len([r for r in results if r.get("success", False)])
            total_time = sum([r.get("processing_metrics", {}).get("total_time", 0) for r in results])
            
            return {
                "success": True,
                "batch_summary": {
                    "total_documents": len(results),
                    "successful_documents": successful,
                    "failed_documents": len(results) - successful,
                    "total_processing_time": total_time,
                    "average_processing_time": total_time / len(results) if results else 0
                },
                "results": results
            }
            
        except Exception as e:
            logger.error(f"Error in batch processing: {e}")
            logger.error(traceback.format_exc())
            return {
                "success": False,
                "error_message": f"Internal server error: {str(e)}"
            }, 500

@ns.route('/supported-entities')
class SupportedEntities(Resource):
    def get(self):
        """Get list of supported PII entity types"""
        try:
            init_processor()
            
            # Standard entity types supported by the system
            entities = {
                "standard": [
                    "PERSON", "EMAIL_ADDRESS", "PHONE_NUMBER", "CREDIT_CARD",
                    "SSN", "IP_ADDRESS", "URL", "ORGANIZATION", "LOCATION",
                    "DATE", "TIME", "FINANCIAL"
                ],
                "security_specific": [
                    "EMPLOYEE_ID", "ACCESS_CARD", "POLICY_NUMBER", "SYSTEM_ID",
                    "BADGE_NUMBER", "AUDIT_REF", "CREDENTIAL_ID", "ROOM_NUMBER",
                    "MAC_ADDRESS"
                ],
                "engines": [
                    "presidio", "spacy", "transformers", "custom_patterns"
                ]
            }
            
            return {
                "success": True,
                "supported_entities": entities
            }
            
        except Exception as e:
            logger.error(f"Error getting supported entities: {e}")
            return {
                "success": False,
                "error_message": str(e)
            }, 500

@ns.route('/logo-classes')
class LogoClasses(Resource):
    def get(self):
        """Get list of supported logo classes"""
        try:
            init_processor()
            
            # Mock logo classes - in production, this would come from the YOLO model
            logo_classes = [
                "microsoft", "google", "apple", "amazon", "facebook",
                "twitter", "linkedin", "generic_company", "government_seal",
                "bank_logo"
            ]
            
            return {
                "success": True,
                "supported_logo_classes": logo_classes
            }
            
        except Exception as e:
            logger.error(f"Error getting logo classes: {e}")
            return {
                "success": False,
                "error_message": str(e)
            }, 500

# File upload endpoint for multipart/form-data
@app.route('/api/v1/upload', methods=['POST'])
def upload_file():
    """Upload file endpoint for multipart/form-data"""
    try:
        # Initialize processor if needed
        init_processor()
        
        if 'file' not in request.files:
            return jsonify({"success": False, "error_message": "No file provided"}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({"success": False, "error_message": "No file selected"}), 400
        
        # Read file
        file_bytes = file.read()
        
        # Convert to image
        pil_image = Image.open(io.BytesIO(file_bytes))
        image_array = np.array(pil_image)
        
        # Convert RGB to BGR for OpenCV
        if len(image_array.shape) == 3 and image_array.shape[2] == 3:
            image_array = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
        
        # Get processing options from form data
        options = {}
        if 'options' in request.form:
            options = json.loads(request.form['options'])
        
        processing_config = create_processing_config(options)
        
        # Process document
        result = processor.process_document(
            image_array,
            document_id=secure_filename(file.filename),
            processing_options=processing_config
        )
        
        # Format response
        response = format_processing_result(result)
        
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Error in file upload: {e}")
        logger.error(traceback.format_exc())
        return jsonify({
            "success": False,
            "error_message": f"Internal server error: {str(e)}"
        }), 500

# Error handlers
@app.errorhandler(404)
def not_found(error):
    return jsonify({"error": "Endpoint not found"}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({"error": "Internal server error"}), 500

@app.errorhandler(413)
def too_large(error):
    return jsonify({"error": "File too large"}), 413

# Main application entry point
def create_app(config_override=None):
    """Create Flask application with configuration"""
    
    # Configure app
    app.config['MAX_CONTENT_LENGTH'] = api_config.max_file_size
    app.config['JSON_SORT_KEYS'] = False
    
    if config_override:
        app.config.update(config_override)
    
    return app

def main():
    """Main function to run the API server"""
    
    # Get configuration
    config = get_api_config()
    
    # Create app
    application = create_app()
    
    # Initialize processor
    init_processor()
    
    logger.info(f"Starting SecureDoc AI API server on {config.host}:{config.port}")
    logger.info(f"Debug mode: {config.debug}")
    logger.info(f"API documentation available at: http://{config.host}:{config.port}/docs/")
    
    # Run server
    application.run(
        host=config.host,
        port=config.port,
        debug=config.debug,
        threaded=True
    )

if __name__ == '__main__':
    main()