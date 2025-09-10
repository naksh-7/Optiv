# API Documentation for SecureDoc AI

## Overview

SecureDoc AI provides a comprehensive REST API for document processing, PII detection, and logo removal. The API is built with Flask and provides both single document and batch processing capabilities.

## Base URL

```
Production: https://api.securedoc-ai.com
Development: http://localhost:8000
```

## Authentication

Currently, the API supports bearer token authentication (optional, configurable):

```bash
curl -H "Authorization: Bearer YOUR_TOKEN" \
     -X POST http://localhost:8000/api/v1/process
```

## Endpoints

### Health Check

**GET** `/api/v1/health`

Returns the health status of the API and all underlying services.

**Response:**
```json
{
  "status": "healthy",
  "processor_initialized": true,
  "timestamp": "2025-01-15T10:30:00Z"
}
```

### Process Document

**POST** `/api/v1/process`

Process a single document for PII detection and logo removal.

**Request Body:**
```json
{
  "image": "base64_encoded_image_data",
  "filename": "document.jpg",
  "options": {
    "ocr_confidence": 0.7,
    "pii_confidence": 0.8,
    "logo_confidence": 0.6,
    "pii_engines": ["presidio", "spacy", "custom_patterns"],
    "logo_removal_method": "inpaint",
    "preserve_format": true,
    "realistic_replacements": true
  }
}
```

**Response:**
```json
{
  "success": true,
  "document_id": "doc_12345",
  "extracted_text": "Original extracted text...",
  "anonymized_text": "Anonymized text with PII removed...",
  "entities": [
    {
      "text": "John Doe",
      "label": "PERSON", 
      "start": 0,
      "end": 8,
      "confidence": 0.95,
      "engine": "spacy",
      "is_sensitive": true
    }
  ],
  "logos": [
    {
      "class_name": "microsoft",
      "confidence": 0.87,
      "bbox": [100, 200, 300, 400],
      "area": 40000
    }
  ],
  "processing_metrics": {
    "total_time": 5.2,
    "ocr_time": 2.1,
    "pii_detection_time": 1.8,
    "logo_detection_time": 0.9,
    "anonymization_time": 0.4,
    "confidence_scores": {
      "ocr_confidence": 0.89,
      "pii_confidence": 0.91,
      "logo_confidence": 0.87
    },
    "entities_found": 5,
    "logos_found": 1
  },
  "anonymization_mapping": {
    "John Doe": "[PERSON]",
    "john.doe@company.com": "user@example.com"
  },
  "processed_image_base64": "base64_encoded_processed_image"
}
```

### Batch Process Documents

**POST** `/api/v1/batch-process`

Process multiple documents in a single request.

**Request Body:**
```json
{
  "images": [
    {
      "image": "base64_encoded_image_1",
      "filename": "doc1.jpg"
    },
    {
      "image": "base64_encoded_image_2", 
      "filename": "doc2.jpg"
    }
  ],
  "options": {
    "ocr_confidence": 0.7,
    "pii_confidence": 0.8
  }
}
```

**Response:**
```json
{
  "success": true,
  "batch_summary": {
    "total_documents": 2,
    "successful_documents": 2,
    "failed_documents": 0,
    "total_processing_time": 8.5,
    "average_processing_time": 4.25
  },
  "results": [
    {
      "success": true,
      "document_id": "doc1.jpg",
      "extracted_text": "...",
      "entities": [...]
    },
    {
      "success": true,
      "document_id": "doc2.jpg", 
      "extracted_text": "...",
      "entities": [...]
    }
  ]
}
```

### File Upload

**POST** `/api/v1/upload`

Upload a file using multipart/form-data (alternative to base64 encoding).

**Request:**
```bash
curl -X POST \
  -F "file=@document.jpg" \
  -F "options={\"ocr_confidence\": 0.8}" \
  http://localhost:8000/api/v1/upload
```

**Response:** Same as `/process` endpoint.

### Get Supported Entities

**GET** `/api/v1/supported-entities`

Returns list of supported PII entity types.

**Response:**
```json
{
  "success": true,
  "supported_entities": {
    "standard": [
      "PERSON", "EMAIL_ADDRESS", "PHONE_NUMBER", 
      "CREDIT_CARD", "SSN", "IP_ADDRESS"
    ],
    "security_specific": [
      "EMPLOYEE_ID", "ACCESS_CARD", "POLICY_NUMBER",
      "SYSTEM_ID", "BADGE_NUMBER"
    ],
    "engines": [
      "presidio", "spacy", "transformers", "custom_patterns"
    ]
  }
}
```

### Get Logo Classes

**GET** `/api/v1/logo-classes`

Returns list of supported logo classes for detection.

**Response:**
```json
{
  "success": true,
  "supported_logo_classes": [
    "microsoft", "google", "apple", "amazon",
    "facebook", "twitter", "linkedin"
  ]
}
```

## Configuration Options

### OCR Options
- `ocr_confidence` (float): Minimum confidence threshold for text extraction (0-1, default: 0.7)

### PII Detection Options  
- `pii_confidence` (float): Minimum confidence threshold for PII detection (0-1, default: 0.8)
- `pii_engines` (array): List of PII detection engines to use
  - Available: `["presidio", "spacy", "transformers", "custom_patterns"]`

### Logo Detection Options
- `logo_confidence` (float): Minimum confidence threshold for logo detection (0-1, default: 0.6)
- `logo_removal_method` (string): Method for removing detected logos
  - Available: `"inpaint"`, `"blur"`, `"pixelate"`, `"black_box"`

### Anonymization Options
- `preserve_format` (boolean): Whether to preserve original text formatting (default: true)
- `realistic_replacements` (boolean): Use realistic replacement values (default: true)

## Error Responses

All error responses follow this format:

```json
{
  "success": false,
  "error_message": "Description of the error",
  "error_code": "ERROR_CODE",
  "timestamp": "2025-01-15T10:30:00Z"
}
```

### Common Error Codes

- `INVALID_IMAGE`: Invalid or corrupted image data
- `PROCESSING_FAILED`: Document processing failed
- `FILE_TOO_LARGE`: Uploaded file exceeds size limit
- `UNSUPPORTED_FORMAT`: Unsupported file format
- `RATE_LIMIT_EXCEEDED`: Too many requests
- `AUTHENTICATION_FAILED`: Invalid authentication token

## Rate Limiting

The API implements rate limiting to ensure fair usage:

- **Default Limit**: 60 requests per minute per IP
- **Burst Allowance**: Up to 10 additional requests in short bursts
- **Headers**: Rate limit information is returned in response headers:
  - `X-RateLimit-Limit`: Maximum requests per window
  - `X-RateLimit-Remaining`: Remaining requests in current window
  - `X-RateLimit-Reset`: Timestamp when the limit resets

## File Size Limits

- **Maximum file size**: 50MB per image
- **Batch processing**: Maximum 20 documents per batch request
- **Total batch size**: Maximum 500MB combined

## Performance Considerations

### Processing Times
- **Single document**: 2-10 seconds depending on size and complexity
- **Batch processing**: Parallel processing reduces total time
- **Large images**: Images over 10MB may take longer to process

### Optimization Tips
1. **Resize images** before upload to reduce processing time
2. **Use appropriate confidence thresholds** - lower values increase processing time
3. **Batch requests** when processing multiple documents
4. **Choose optimal logo removal method**:
   - `inpaint`: Highest quality, slowest
   - `blur`: Good quality, fast
   - `black_box`: Lowest quality, fastest

## SDK and Libraries

### Python SDK

```python
import requests
import base64

class SecureDocClient:
    def __init__(self, base_url, api_key=None):
        self.base_url = base_url
        self.api_key = api_key
    
    def process_document(self, image_path, options=None):
        with open(image_path, 'rb') as f:
            image_data = base64.b64encode(f.read()).decode()
        
        payload = {
            'image': image_data,
            'filename': image_path,
            'options': options or {}
        }
        
        headers = {'Content-Type': 'application/json'}
        if self.api_key:
            headers['Authorization'] = f'Bearer {self.api_key}'
        
        response = requests.post(
            f'{self.base_url}/api/v1/process',
            json=payload,
            headers=headers
        )
        
        return response.json()

# Usage
client = SecureDocClient('http://localhost:8000')
result = client.process_document('document.jpg', {
    'pii_confidence': 0.8,
    'logo_removal_method': 'inpaint'
})
```

### cURL Examples

```bash
# Process single document
curl -X POST http://localhost:8000/api/v1/process \
  -H "Content-Type: application/json" \
  -d '{
    "image": "'$(base64 -i document.jpg)'",
    "options": {"pii_confidence": 0.8}
  }'

# Upload file
curl -X POST http://localhost:8000/api/v1/upload \
  -F "file=@document.jpg" \
  -F 'options={"logo_removal_method": "blur"}'

# Health check
curl http://localhost:8000/api/v1/health
```

## Webhooks (Coming Soon)

Support for webhook notifications when processing is complete:

```json
{
  "event": "processing_complete",
  "document_id": "doc_12345",
  "status": "success",
  "processing_time": 5.2,
  "webhook_url": "https://your-app.com/webhook"
}
```

## Support

For API support and questions:
- **Documentation**: https://docs.securedoc-ai.com
- **GitHub Issues**: https://github.com/yourusername/securedoc-ai/issues
- **Email**: support@securedoc-ai.com