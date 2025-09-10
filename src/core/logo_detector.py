# YOLO-based Logo Detection and Removal System
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image, ImageDraw
import torch
from typing import List, Dict, Tuple, Optional, Union
import logging
from dataclasses import dataclass
from pathlib import Path
import json
import yaml

logger = logging.getLogger(__name__)

@dataclass
class LogoDetection:
    """Logo detection result"""
    bbox: Tuple[int, int, int, int]  # (x1, y1, x2, y2)
    confidence: float
    class_id: int
    class_name: str
    area: float

@dataclass 
class LogoRemovalResult:
    """Logo removal operation result"""
    original_detections: List[LogoDetection]
    processed_image: np.ndarray
    removal_method: str
    processing_time: float
    success: bool
    removed_logos: int

class LogoClassifier:
    """Logo classification and management"""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path or "config/logo_classes.yaml"
        self.logo_classes = self._load_logo_classes()
    
    def _load_logo_classes(self) -> Dict[str, Dict]:
        """Load logo class definitions"""
        try:
            if Path(self.config_path).exists():
                with open(self.config_path, 'r') as f:
                    return yaml.safe_load(f)
            else:
                # Default logo classes for common corporate logos
                default_classes = {
                    0: {"name": "microsoft", "category": "tech", "sensitive": True},
                    1: {"name": "google", "category": "tech", "sensitive": True},
                    2: {"name": "apple", "category": "tech", "sensitive": True},
                    3: {"name": "amazon", "category": "tech", "sensitive": True},
                    4: {"name": "facebook", "category": "social", "sensitive": True},
                    5: {"name": "twitter", "category": "social", "sensitive": True},
                    6: {"name": "linkedin", "category": "social", "sensitive": True},
                    7: {"name": "generic_company", "category": "business", "sensitive": True},
                    8: {"name": "government_seal", "category": "government", "sensitive": True},
                    9: {"name": "bank_logo", "category": "financial", "sensitive": True}
                }
                
                # Save default config
                Path(self.config_path).parent.mkdir(parents=True, exist_ok=True)
                with open(self.config_path, 'w') as f:
                    yaml.dump(default_classes, f)
                
                return default_classes
        
        except Exception as e:
            logger.error(f"Error loading logo classes: {e}")
            return {}
    
    def get_class_name(self, class_id: int) -> str:
        """Get class name by ID"""
        return self.logo_classes.get(class_id, {}).get("name", f"unknown_{class_id}")
    
    def is_sensitive_logo(self, class_id: int) -> bool:
        """Check if logo class is sensitive and should be removed"""
        return self.logo_classes.get(class_id, {}).get("sensitive", False)

class YOLOLogoDetector:
    """YOLO-based logo detection system"""
    
    def __init__(self, 
                 model_path: str = "yolov8n.pt",
                 confidence_threshold: float = 0.6,
                 iou_threshold: float = 0.5,
                 device: str = "auto"):
        
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.device = self._setup_device(device)
        
        # Initialize YOLO model
        self.model = self._load_model()
        self.classifier = LogoClassifier()
    
    def _setup_device(self, device: str) -> str:
        """Setup computation device"""
        if device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return device
    
    def _load_model(self) -> YOLO:
        """Load YOLO model"""
        try:
            # Check if custom trained model exists
            custom_model_path = Path("models") / "logo_detection" / "best.pt"
            
            if custom_model_path.exists():
                logger.info(f"Loading custom logo detection model: {custom_model_path}")
                model = YOLO(str(custom_model_path))
            else:
                # Use pre-trained YOLO model (will need fine-tuning for logos)
                logger.info(f"Loading pre-trained YOLO model: {self.model_path}")
                model = YOLO(self.model_path)
            
            # Move to device
            model.to(self.device)
            return model
            
        except Exception as e:
            logger.error(f"Error loading YOLO model: {e}")
            raise
    
    def detect_logos(self, image: Union[np.ndarray, str, Path]) -> List[LogoDetection]:
        """Detect logos in image"""
        try:
            # Convert image if needed
            if isinstance(image, (str, Path)):
                image = cv2.imread(str(image))
            
            if image is None:
                raise ValueError("Invalid image provided")
            
            # Run YOLO inference
            results = self.model(
                image,
                conf=self.confidence_threshold,
                iou=self.iou_threshold,
                verbose=False
            )
            
            detections = []
            
            # Process results
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for i in range(len(boxes)):
                        # Extract bounding box
                        box = boxes.xyxy[i].cpu().numpy()
                        x1, y1, x2, y2 = map(int, box)
                        
                        # Extract confidence and class
                        confidence = float(boxes.conf[i].cpu().numpy())
                        class_id = int(boxes.cls[i].cpu().numpy())
                        class_name = self.classifier.get_class_name(class_id)
                        
                        # Calculate area
                        area = (x2 - x1) * (y2 - y1)
                        
                        detections.append(LogoDetection(
                            bbox=(x1, y1, x2, y2),
                            confidence=confidence,
                            class_id=class_id,
                            class_name=class_name,
                            area=area
                        ))
            
            # Sort by confidence
            detections.sort(key=lambda x: x.confidence, reverse=True)
            
            logger.info(f"Detected {len(detections)} logos")
            return detections
            
        except Exception as e:
            logger.error(f"Error in logo detection: {e}")
            return []

class LogoRemover:
    """Advanced logo removal system"""
    
    def __init__(self, detector: YOLOLogoDetector):
        self.detector = detector
    
    def remove_logos(self, 
                    image: Union[np.ndarray, str, Path],
                    method: str = "inpaint",
                    selective: bool = True,
                    min_confidence: float = 0.7) -> LogoRemovalResult:
        """Remove detected logos from image"""
        
        import time
        start_time = time.time()
        
        try:
            # Load image if path provided
            if isinstance(image, (str, Path)):
                original_image = cv2.imread(str(image))
            else:
                original_image = image.copy()
            
            if original_image is None:
                raise ValueError("Invalid image provided")
            
            # Detect logos
            detections = self.detector.detect_logos(original_image)
            
            # Filter detections
            filtered_detections = []
            for detection in detections:
                # Apply confidence filter
                if detection.confidence < min_confidence:
                    continue
                
                # Apply selective filtering (only sensitive logos)
                if selective and not self.detector.classifier.is_sensitive_logo(detection.class_id):
                    continue
                
                filtered_detections.append(detection)
            
            if not filtered_detections:
                return LogoRemovalResult(
                    original_detections=detections,
                    processed_image=original_image,
                    removal_method=method,
                    processing_time=time.time() - start_time,
                    success=True,
                    removed_logos=0
                )
            
            # Apply removal method
            processed_image = self._apply_removal_method(
                original_image, 
                filtered_detections, 
                method
            )
            
            return LogoRemovalResult(
                original_detections=detections,
                processed_image=processed_image,
                removal_method=method,
                processing_time=time.time() - start_time,
                success=True,
                removed_logos=len(filtered_detections)
            )
            
        except Exception as e:
            logger.error(f"Error in logo removal: {e}")
            return LogoRemovalResult(
                original_detections=[],
                processed_image=original_image if 'original_image' in locals() else image,
                removal_method=method,
                processing_time=time.time() - start_time,
                success=False,
                removed_logos=0
            )
    
    def _apply_removal_method(self, 
                             image: np.ndarray, 
                             detections: List[LogoDetection], 
                             method: str) -> np.ndarray:
        """Apply specific logo removal method"""
        
        processed = image.copy()
        
        if method == "inpaint":
            processed = self._inpaint_logos(processed, detections)
        elif method == "blur":
            processed = self._blur_logos(processed, detections)
        elif method == "pixelate":
            processed = self._pixelate_logos(processed, detections)
        elif method == "black_box":
            processed = self._black_box_logos(processed, detections)
        elif method == "replace":
            processed = self._replace_logos(processed, detections)
        else:
            logger.warning(f"Unknown removal method: {method}, using black_box")
            processed = self._black_box_logos(processed, detections)
        
        return processed
    
    def _inpaint_logos(self, image: np.ndarray, detections: List[LogoDetection]) -> np.ndarray:
        """Remove logos using OpenCV inpainting"""
        # Create mask for inpainting
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        
        for detection in detections:
            x1, y1, x2, y2 = detection.bbox
            # Expand region slightly for better inpainting
            padding = 5
            x1 = max(0, x1 - padding)
            y1 = max(0, y1 - padding)
            x2 = min(image.shape[1], x2 + padding)
            y2 = min(image.shape[0], y2 + padding)
            
            cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)
        
        # Apply inpainting
        result = cv2.inpaint(image, mask, 3, cv2.INPAINT_TELEA)
        return result
    
    def _blur_logos(self, image: np.ndarray, detections: List[LogoDetection]) -> np.ndarray:
        """Remove logos using Gaussian blur"""
        result = image.copy()
        
        for detection in detections:
            x1, y1, x2, y2 = detection.bbox
            
            # Extract region
            region = result[y1:y2, x1:x2]
            
            # Apply strong blur
            blurred_region = cv2.GaussianBlur(region, (51, 51), 0)
            
            # Replace region
            result[y1:y2, x1:x2] = blurred_region
        
        return result
    
    def _pixelate_logos(self, image: np.ndarray, detections: List[LogoDetection]) -> np.ndarray:
        """Remove logos using pixelation"""
        result = image.copy()
        
        for detection in detections:
            x1, y1, x2, y2 = detection.bbox
            
            # Extract region
            region = result[y1:y2, x1:x2]
            
            # Pixelate by downsampling and upsampling
            h, w = region.shape[:2]
            pixel_size = 20
            
            # Downsample
            small = cv2.resize(region, (w // pixel_size, h // pixel_size), interpolation=cv2.INTER_LINEAR)
            
            # Upsample back
            pixelated = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)
            
            # Replace region
            result[y1:y2, x1:x2] = pixelated
        
        return result
    
    def _black_box_logos(self, image: np.ndarray, detections: List[LogoDetection]) -> np.ndarray:
        """Remove logos using black rectangles"""
        result = image.copy()
        
        for detection in detections:
            x1, y1, x2, y2 = detection.bbox
            cv2.rectangle(result, (x1, y1), (x2, y2), (0, 0, 0), -1)
        
        return result
    
    def _replace_logos(self, image: np.ndarray, detections: List[LogoDetection]) -> np.ndarray:
        """Replace logos with neutral patterns"""
        result = image.copy()
        
        for detection in detections:
            x1, y1, x2, y2 = detection.bbox
            
            # Calculate average color of surrounding area
            padding = 20
            x1_pad = max(0, x1 - padding)
            y1_pad = max(0, y1 - padding)
            x2_pad = min(image.shape[1], x2 + padding)
            y2_pad = min(image.shape[0], y2 + padding)
            
            # Get surrounding region (excluding the logo area)
            surrounding = result[y1_pad:y2_pad, x1_pad:x2_pad].copy()
            surrounding[y1-y1_pad:y2-y1_pad, x1-x1_pad:x2-x1_pad] = 0
            
            # Calculate average color
            mask = np.any(surrounding != 0, axis=2)
            if np.any(mask):
                avg_color = np.mean(surrounding[mask], axis=0)
            else:
                avg_color = [128, 128, 128]  # Gray fallback
            
            # Fill logo area with average color
            cv2.rectangle(result, (x1, y1), (x2, y2), avg_color, -1)
        
        return result
    
    def visualize_detections(self, 
                           image: np.ndarray, 
                           detections: List[LogoDetection],
                           save_path: Optional[str] = None) -> np.ndarray:
        """Visualize logo detections on image"""
        
        vis_image = image.copy()
        
        # Color map for different classes
        colors = [
            (0, 255, 0),    # Green
            (255, 0, 0),    # Blue
            (0, 0, 255),    # Red
            (255, 255, 0),  # Cyan
            (255, 0, 255),  # Magenta
            (0, 255, 255),  # Yellow
        ]
        
        for i, detection in enumerate(detections):
            x1, y1, x2, y2 = detection.bbox
            color = colors[detection.class_id % len(colors)]
            
            # Draw bounding box
            cv2.rectangle(vis_image, (x1, y1), (x2, y2), color, 2)
            
            # Draw label
            label = f"{detection.class_name}: {detection.confidence:.2f}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
            
            # Draw label background
            cv2.rectangle(
                vis_image,
                (x1, y1 - label_size[1] - 10),
                (x1 + label_size[0], y1),
                color,
                -1
            )
            
            # Draw label text
            cv2.putText(
                vis_image,
                label,
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1
            )
        
        if save_path:
            cv2.imwrite(save_path, vis_image)
        
        return vis_image

class LogoDetectionPipeline:
    """Complete logo detection and removal pipeline"""
    
    def __init__(self, config: Dict[str, any] = None):
        if config is None:
            config = {
                "model_path": "yolov8n.pt",
                "confidence_threshold": 0.6,
                "iou_threshold": 0.5,
                "device": "auto"
            }
        
        self.detector = YOLOLogoDetector(**config)
        self.remover = LogoRemover(self.detector)
    
    def process_document(self, 
                        image: Union[np.ndarray, str, Path],
                        remove_logos: bool = True,
                        removal_method: str = "inpaint",
                        visualize: bool = False,
                        output_dir: Optional[str] = None) -> Dict[str, any]:
        """Process document for logo detection and removal"""
        
        results = {
            "detections": [],
            "removal_result": None,
            "visualization": None,
            "success": False
        }
        
        try:
            # Detect logos
            detections = self.detector.detect_logos(image)
            results["detections"] = detections
            
            if remove_logos and detections:
                # Remove logos
                removal_result = self.remover.remove_logos(
                    image, 
                    method=removal_method
                )
                results["removal_result"] = removal_result
                results["processed_image"] = removal_result.processed_image
            else:
                # No removal needed
                if isinstance(image, (str, Path)):
                    results["processed_image"] = cv2.imread(str(image))
                else:
                    results["processed_image"] = image.copy()
            
            # Create visualization if requested
            if visualize:
                vis_image = self.remover.visualize_detections(
                    results["processed_image"] if remove_logos else results["processed_image"],
                    detections
                )
                results["visualization"] = vis_image
            
            results["success"] = True
            
            # Save outputs if directory specified
            if output_dir:
                output_path = Path(output_dir)
                output_path.mkdir(parents=True, exist_ok=True)
                
                # Save processed image
                cv2.imwrite(
                    str(output_path / "processed_image.jpg"),
                    results["processed_image"]
                )
                
                # Save visualization if created
                if visualize:
                    cv2.imwrite(
                        str(output_path / "detections_visualization.jpg"),
                        results["visualization"]
                    )
                
                # Save detection results
                detection_data = [
                    {
                        "bbox": det.bbox,
                        "confidence": det.confidence,
                        "class_name": det.class_name,
                        "area": det.area
                    }
                    for det in detections
                ]
                
                with open(output_path / "detections.json", 'w') as f:
                    json.dump(detection_data, f, indent=2)
            
        except Exception as e:
            logger.error(f"Error in logo detection pipeline: {e}")
            results["error"] = str(e)
        
        return results

# Example usage
if __name__ == "__main__":
    # Initialize pipeline
    pipeline = LogoDetectionPipeline({
        "confidence_threshold": 0.6,
        "device": "auto"
    })
    
    # Test with sample image
    test_image = "test_document_with_logos.jpg"
    
    if Path(test_image).exists():
        results = pipeline.process_document(
            test_image,
            remove_logos=True,
            removal_method="inpaint",
            visualize=True,
            output_dir="logo_detection_results"
        )
        
        print(f"Detected {len(results['detections'])} logos")
        if results["removal_result"]:
            print(f"Removed {results['removal_result'].removed_logos} logos")
        
    else:
        print("No test image found. Place test image with logos to test the system.")