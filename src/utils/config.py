# Configuration Management for SecureDoc AI
from typing import Any, Dict, Optional
from pathlib import Path
import yaml
from pydantic import BaseModel, Field
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class OCRConfig(BaseModel):
    """OCR Engine Configuration"""
    primary_engine: str = Field(default="easyocr", description="Primary OCR engine")
    fallback_engine: str = Field(default="tesseract", description="Fallback OCR engine")
    confidence_threshold: float = Field(default=0.7, description="Minimum confidence for text extraction")
    languages: list[str] = Field(default=["en"], description="Supported languages")
    preprocessing: Dict[str, Any] = Field(default_factory=dict, description="Image preprocessing options")

class PIIConfig(BaseModel):
    """PII Detection Configuration"""
    engines: list[str] = Field(default=["presidio", "spacy", "transformers"], description="PII detection engines")
    confidence_threshold: float = Field(default=0.8, description="Minimum confidence for PII detection")
    entity_types: list[str] = Field(
        default=[
            "PERSON", "EMAIL_ADDRESS", "PHONE_NUMBER", "CREDIT_CARD", 
            "SSN", "IP_ADDRESS", "URL", "EMPLOYEE_ID", "ACCESS_CARD"
        ],
        description="Entity types to detect"
    )
    custom_patterns_enabled: bool = Field(default=True, description="Enable custom pattern recognition")
    context_enhancement: bool = Field(default=True, description="Enable context-aware enhancement")

class LogoDetectionConfig(BaseModel):
    """Logo Detection Configuration"""
    model_path: str = Field(default="yolov8n.pt", description="YOLO model path")
    confidence_threshold: float = Field(default=0.6, description="Minimum confidence for logo detection")
    iou_threshold: float = Field(default=0.5, description="IoU threshold for NMS")
    max_detections: int = Field(default=100, description="Maximum detections per image")
    logo_classes: list[str] = Field(default_factory=list, description="Logo classes to detect")

class AnonymizationConfig(BaseModel):
    """Anonymization Configuration"""
    default_strategy: str = Field(default="replace", description="Default masking strategy")
    strategies: Dict[str, Dict[str, Any]] = Field(
        default_factory=lambda: {
            "replace": {"replacement_text": "[REDACTED]"},
            "mask": {"masking_char": "*"},
            "blur": {"blur_radius": 10},
            "encrypt": {"algorithm": "sha256"}
        },
        description="Available masking strategies"
    )
    preserve_format: bool = Field(default=True, description="Preserve original text formatting")

class APIConfig(BaseModel):
    """API Configuration"""
    host: str = Field(default="0.0.0.0", description="API host")
    port: int = Field(default=8000, description="API port")
    debug: bool = Field(default=False, description="Debug mode")
    cors_enabled: bool = Field(default=True, description="Enable CORS")
    rate_limiting: bool = Field(default=True, description="Enable rate limiting")
    max_file_size: int = Field(default=50 * 1024 * 1024, description="Maximum file size (50MB)")

class StreamlitConfig(BaseModel):
    """Streamlit Configuration"""
    page_title: str = Field(default="SecureDoc AI", description="Page title")
    page_icon: str = Field(default="🔒", description="Page icon")
    layout: str = Field(default="wide", description="Page layout")
    theme: Dict[str, str] = Field(
        default_factory=lambda: {
            "primaryColor": "#FF6B6B",
            "backgroundColor": "#FFFFFF",
            "secondaryBackgroundColor": "#F0F2F6",
            "textColor": "#262730"
        },
        description="Streamlit theme"
    )

class LoggingConfig(BaseModel):
    """Logging Configuration"""
    level: str = Field(default="INFO", description="Logging level")
    format: str = Field(
        default="{time:YYYY-MM-DD HH:mm:ss} | {level} | {name}:{function}:{line} | {message}",
        description="Log format"
    )
    file_path: Optional[str] = Field(default=None, description="Log file path")
    max_file_size: str = Field(default="10 MB", description="Maximum log file size")
    retention: str = Field(default="30 days", description="Log retention period")

class ModelConfig(BaseModel):
    """Model Configuration"""
    cache_dir: str = Field(default="models", description="Model cache directory")
    auto_download: bool = Field(default=True, description="Auto-download missing models")
    spacy_model: str = Field(default="en_core_web_lg", description="spaCy model name")
    transformers_model: str = Field(default="bert-base-uncased", description="Transformers model name")
    yolo_model: str = Field(default="yolov8n.pt", description="YOLO model path")

class Config(BaseModel):
    """Main Configuration Class"""
    ocr: OCRConfig = Field(default_factory=OCRConfig)
    pii: PIIConfig = Field(default_factory=PIIConfig)
    logo_detection: LogoDetectionConfig = Field(default_factory=LogoDetectionConfig)
    anonymization: AnonymizationConfig = Field(default_factory=AnonymizationConfig)
    api: APIConfig = Field(default_factory=APIConfig)
    streamlit: StreamlitConfig = Field(default_factory=StreamlitConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    models: ModelConfig = Field(default_factory=ModelConfig)

class ConfigManager:
    """Configuration Manager for SecureDoc AI"""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path or self._get_default_config_path()
        self._config: Optional[Config] = None
    
    def _get_default_config_path(self) -> str:
        """Get default configuration path"""
        # Try environment variable first
        if config_path := os.getenv("SECUREDOC_CONFIG"):
            return config_path
        
        # Look for config in standard locations
        possible_paths = [
            "config/config.yaml",
            "config.yaml",
            os.path.expanduser("~/.securedoc/config.yaml"),
            "/etc/securedoc/config.yaml"
        ]
        
        for path in possible_paths:
            if Path(path).exists():
                return path
        
        # Return default path (will create if not exists)
        return "config/config.yaml"
    
    def load_config(self) -> Config:
        """Load configuration from file"""
        if self._config is not None:
            return self._config
        
        config_file = Path(self.config_path)
        
        if config_file.exists():
            try:
                with open(config_file, 'r', encoding='utf-8') as f:
                    config_data = yaml.safe_load(f)
                
                # Merge with environment variables
                config_data = self._merge_env_vars(config_data)
                
                self._config = Config.model_validate(config_data)
            except Exception as e:
                print(f"Error loading config from {config_file}: {e}")
                print("Using default configuration")
                self._config = Config()
        else:
            print(f"Config file not found at {config_file}, using defaults")
            self._config = Config()
            # Create default config file
            self.save_config(self._config)
        
        return self._config
    
    def _merge_env_vars(self, config_data: Dict[str, Any]) -> Dict[str, Any]:
        """Merge environment variables into configuration"""
        # Environment variable mappings
        env_mappings = {
            "SECUREDOC_API_HOST": ("api", "host"),
            "SECUREDOC_API_PORT": ("api", "port"),
            "SECUREDOC_DEBUG": ("api", "debug"),
            "SECUREDOC_LOG_LEVEL": ("logging", "level"),
            "SECUREDOC_MODEL_CACHE_DIR": ("models", "cache_dir"),
            "SECUREDOC_OCR_CONFIDENCE": ("ocr", "confidence_threshold"),
            "SECUREDOC_PII_CONFIDENCE": ("pii", "confidence_threshold"),
            "SECUREDOC_LOGO_CONFIDENCE": ("logo_detection", "confidence_threshold"),
        }
        
        for env_var, (section, key) in env_mappings.items():
            if value := os.getenv(env_var):
                if section not in config_data:
                    config_data[section] = {}
                
                # Type conversion
                if key in ["port", "max_detections", "max_file_size"]:
                    config_data[section][key] = int(value)
                elif key in ["confidence_threshold", "iou_threshold"]:
                    config_data[section][key] = float(value)
                elif key in ["debug", "cors_enabled", "rate_limiting", "auto_download"]:
                    config_data[section][key] = value.lower() in ("true", "1", "yes", "on")
                else:
                    config_data[section][key] = value
        
        return config_data
    
    def save_config(self, config: Config) -> None:
        """Save configuration to file"""
        config_file = Path(self.config_path)
        config_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(config_file, 'w', encoding='utf-8') as f:
            yaml.dump(
                config.model_dump(exclude_none=True),
                f,
                default_flow_style=False,
                sort_keys=False,
                indent=2
            )
    
    def get_config(self) -> Config:
        """Get current configuration"""
        return self.load_config()
    
    def reload_config(self) -> Config:
        """Reload configuration from file"""
        self._config = None
        return self.load_config()
    
    def update_config(self, updates: Dict[str, Any]) -> Config:
        """Update configuration with new values"""
        config = self.get_config()
        
        # Deep update configuration
        for key, value in updates.items():
            if hasattr(config, key) and isinstance(getattr(config, key), BaseModel):
                # Update nested config
                nested_config = getattr(config, key)
                for nested_key, nested_value in value.items():
                    setattr(nested_config, nested_key, nested_value)
            else:
                # Update top-level config
                setattr(config, key, value)
        
        self.save_config(config)
        return config

# Global configuration manager instance
config_manager = ConfigManager()

def get_config() -> Config:
    """Get global configuration instance"""
    return config_manager.get_config()

def reload_config() -> Config:
    """Reload global configuration"""
    return config_manager.reload_config()

# Export commonly used configurations
def get_ocr_config() -> OCRConfig:
    return get_config().ocr

def get_pii_config() -> PIIConfig:
    return get_config().pii

def get_logo_config() -> LogoDetectionConfig:
    return get_config().logo_detection

def get_api_config() -> APIConfig:
    return get_config().api

def get_streamlit_config() -> StreamlitConfig:
    return get_config().streamlit