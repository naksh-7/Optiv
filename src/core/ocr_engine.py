# Advanced OCR Engine with Confidence Scoring
import cv2
import numpy as np
from PIL import Image
import easyocr
import pytesseract
from typing import List, Dict, Tuple, Optional, Union
import logging
from dataclasses import dataclass
from pathlib import Path
import json
from concurrent.futures import ThreadPoolExecutor
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class OCRResult:
    """OCR Result with confidence scoring"""
    text: str
    confidence: float
    bbox: Tuple[int, int, int, int]  # (x1, y1, x2, y2)
    engine: str
    processing_time: float
    language: str = "en"

@dataclass
class DocumentAnalysis:
    """Complete document analysis result"""
    text_blocks: List[OCRResult]
    full_text: str
    average_confidence: float
    total_processing_time: float
    image_info: Dict[str, any]
    preprocessing_applied: List[str]

class ImagePreprocessor:
    """Advanced image preprocessing for better OCR results"""
    
    @staticmethod
    def enhance_image(image: np.ndarray, config: Dict[str, any] = None) -> Tuple[np.ndarray, List[str]]:
        """Apply image enhancements for better OCR"""
        if config is None:
            config = {
                "denoise": True,
                "deskew": True,
                "enhance_contrast": True,
                "sharpen": True,
                "resize_factor": 1.5
            }
        
        enhanced = image.copy()
        applied_operations = []
        
        try:
            # Convert to grayscale if needed
            if len(enhanced.shape) == 3:
                enhanced = cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY)
                applied_operations.append("grayscale_conversion")
            
            # Resize for better recognition
            if config.get("resize_factor", 1.0) != 1.0:
                factor = config["resize_factor"]
                height, width = enhanced.shape
                enhanced = cv2.resize(enhanced, (int(width * factor), int(height * factor)))
                applied_operations.append(f"resize_{factor}x")
            
            # Denoise
            if config.get("denoise", True):
                enhanced = cv2.fastNlMeansDenoising(enhanced)
                applied_operations.append("denoise")
            
            # Deskew
            if config.get("deskew", True):
                enhanced = ImagePreprocessor._deskew_image(enhanced)
                applied_operations.append("deskew")
            
            # Enhance contrast
            if config.get("enhance_contrast", True):
                enhanced = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8)).apply(enhanced)
                applied_operations.append("enhance_contrast")
            
            # Sharpen
            if config.get("sharpen", True):
                kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
                enhanced = cv2.filter2D(enhanced, -1, kernel)
                applied_operations.append("sharpen")
            
            # Binary threshold
            if config.get("binary_threshold", True):
                _, enhanced = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                applied_operations.append("binary_threshold")
            
        except Exception as e:
            logger.warning(f"Preprocessing error: {e}")
            return image, ["preprocessing_failed"]
        
        return enhanced, applied_operations
    
    @staticmethod
    def _deskew_image(image: np.ndarray) -> np.ndarray:
        """Deskew image using Hough transform"""
        try:
            coords = np.column_stack(np.where(image > 0))
            if len(coords) == 0:
                return image
            
            angle = cv2.minAreaRect(coords)[-1]
            
            if angle < -45:
                angle = -(90 + angle)
            else:
                angle = -angle
            
            # Only apply deskewing if angle is significant
            if abs(angle) > 0.5:
                (h, w) = image.shape[:2]
                center = (w // 2, h // 2)
                M = cv2.getRotationMatrix2D(center, angle, 1.0)
                rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
                return rotated
            
        except Exception:
            pass
        
        return image

class EasyOCREngine:
    """EasyOCR engine wrapper with optimizations"""
    
    def __init__(self, languages: List[str] = ["en"], gpu: bool = True):
        self.reader = easyocr.Reader(languages, gpu=gpu)
        self.languages = languages
    
    def extract_text(self, image: np.ndarray, confidence_threshold: float = 0.7) -> List[OCRResult]:
        """Extract text using EasyOCR"""
        start_time = time.time()
        
        try:
            results = self.reader.readtext(image, detail=1)
            ocr_results = []
            
            for bbox, text, confidence in results:
                if confidence >= confidence_threshold:
                    # Convert bbox to (x1, y1, x2, y2) format
                    bbox_coords = (
                        int(min([point[0] for point in bbox])),
                        int(min([point[1] for point in bbox])),
                        int(max([point[0] for point in bbox])),
                        int(max([point[1] for point in bbox]))
                    )
                    
                    ocr_results.append(OCRResult(
                        text=text.strip(),
                        confidence=confidence,
                        bbox=bbox_coords,
                        engine="easyocr",
                        processing_time=time.time() - start_time,
                        language="en"  # EasyOCR doesn't provide language detection per result
                    ))
            
            return ocr_results
            
        except Exception as e:
            logger.error(f"EasyOCR error: {e}")
            return []

class TesseractEngine:
    """Tesseract OCR engine wrapper"""
    
    def __init__(self, languages: str = "eng", config: str = "--oem 3 --psm 6"):
        self.languages = languages
        self.config = config
    
    def extract_text(self, image: np.ndarray, confidence_threshold: float = 0.7) -> List[OCRResult]:
        """Extract text using Tesseract"""
        start_time = time.time()
        
        try:
            # Get detailed data from Tesseract
            data = pytesseract.image_to_data(
                image, 
                lang=self.languages,
                config=self.config,
                output_type=pytesseract.Output.DICT
            )
            
            ocr_results = []
            n_boxes = len(data['text'])
            
            for i in range(n_boxes):
                confidence = int(data['conf'][i])
                text = data['text'][i].strip()
                
                if confidence >= (confidence_threshold * 100) and text:
                    bbox = (
                        data['left'][i],
                        data['top'][i],
                        data['left'][i] + data['width'][i],
                        data['top'][i] + data['height'][i]
                    )
                    
                    ocr_results.append(OCRResult(
                        text=text,
                        confidence=confidence / 100.0,
                        bbox=bbox,
                        engine="tesseract",
                        processing_time=time.time() - start_time,
                        language=self.languages
                    ))
            
            return ocr_results
            
        except Exception as e:
            logger.error(f"Tesseract error: {e}")
            return []

class HybridOCREngine:
    """Hybrid OCR engine combining multiple engines for optimal results"""
    
    def __init__(self, 
                 primary_engine: str = "easyocr",
                 fallback_engine: str = "tesseract",
                 languages: List[str] = ["en"],
                 gpu: bool = True):
        
        self.primary_engine_name = primary_engine
        self.fallback_engine_name = fallback_engine
        
        # Initialize engines
        if "easyocr" in [primary_engine, fallback_engine]:
            self.easyocr = EasyOCREngine(languages, gpu)
        
        if "tesseract" in [primary_engine, fallback_engine]:
            self.tesseract = TesseractEngine("eng")
        
        self.preprocessor = ImagePreprocessor()
    
    def process_document(self, 
                        image: Union[np.ndarray, str, Path],
                        preprocessing_config: Dict[str, any] = None,
                        confidence_threshold: float = 0.7,
                        use_fallback: bool = True) -> DocumentAnalysis:
        """Process complete document with hybrid OCR approach"""
        
        start_time = time.time()
        
        # Load image if path provided
        if isinstance(image, (str, Path)):
            image = cv2.imread(str(image))
        
        if image is None:
            raise ValueError("Invalid image provided")
        
        # Get image info
        image_info = {
            "original_shape": image.shape,
            "channels": len(image.shape),
            "dtype": str(image.dtype)
        }
        
        # Preprocess image
        processed_image, preprocessing_applied = self.preprocessor.enhance_image(image, preprocessing_config)
        
        # Primary OCR engine
        primary_results = self._extract_with_engine(processed_image, self.primary_engine_name, confidence_threshold)
        
        # Fallback if primary results are poor
        fallback_results = []
        if use_fallback and self._should_use_fallback(primary_results):
            logger.info(f"Using fallback engine: {self.fallback_engine_name}")
            fallback_results = self._extract_with_engine(processed_image, self.fallback_engine_name, confidence_threshold)
        
        # Combine and optimize results
        all_results = self._combine_results(primary_results, fallback_results)
        
        # Generate full text
        full_text = self._generate_full_text(all_results)
        
        # Calculate metrics
        avg_confidence = np.mean([r.confidence for r in all_results]) if all_results else 0.0
        total_time = time.time() - start_time
        
        return DocumentAnalysis(
            text_blocks=all_results,
            full_text=full_text,
            average_confidence=avg_confidence,
            total_processing_time=total_time,
            image_info=image_info,
            preprocessing_applied=preprocessing_applied
        )
    
    def _extract_with_engine(self, image: np.ndarray, engine_name: str, confidence_threshold: float) -> List[OCRResult]:
        """Extract text with specified engine"""
        if engine_name == "easyocr" and hasattr(self, 'easyocr'):
            return self.easyocr.extract_text(image, confidence_threshold)
        elif engine_name == "tesseract" and hasattr(self, 'tesseract'):
            return self.tesseract.extract_text(image, confidence_threshold)
        else:
            logger.warning(f"Engine {engine_name} not available")
            return []
    
    def _should_use_fallback(self, primary_results: List[OCRResult]) -> bool:
        """Determine if fallback engine should be used"""
        if not primary_results:
            return True
        
        avg_confidence = np.mean([r.confidence for r in primary_results])
        total_text = sum(len(r.text) for r in primary_results)
        
        # Use fallback if confidence is low or very little text extracted
        return avg_confidence < 0.8 or total_text < 50
    
    def _combine_results(self, primary: List[OCRResult], fallback: List[OCRResult]) -> List[OCRResult]:
        """Combine results from multiple engines, preferring higher confidence"""
        if not fallback:
            return primary
        
        if not primary:
            return fallback
        
        # Simple combination: use all results, could be enhanced with NMS
        combined = primary + fallback
        
        # Sort by confidence
        combined.sort(key=lambda x: x.confidence, reverse=True)
        
        return combined
    
    def _generate_full_text(self, results: List[OCRResult]) -> str:
        """Generate full text from OCR results"""
        if not results:
            return ""
        
        # Sort by position (top to bottom, left to right)
        sorted_results = sorted(results, key=lambda x: (x.bbox[1], x.bbox[0]))
        
        # Group by lines based on y-coordinate
        lines = []
        current_line = []
        current_y = -1
        line_threshold = 20  # pixels
        
        for result in sorted_results:
            y_center = (result.bbox[1] + result.bbox[3]) // 2
            
            if current_y == -1 or abs(y_center - current_y) <= line_threshold:
                current_line.append(result)
                current_y = y_center
            else:
                if current_line:
                    # Sort current line by x-coordinate
                    current_line.sort(key=lambda x: x.bbox[0])
                    lines.append(current_line)
                current_line = [result]
                current_y = y_center
        
        # Don't forget the last line
        if current_line:
            current_line.sort(key=lambda x: x.bbox[0])
            lines.append(current_line)
        
        # Join text
        full_text_lines = []
        for line in lines:
            line_text = " ".join([result.text for result in line])
            full_text_lines.append(line_text)
        
        return "\n".join(full_text_lines)
    
    def batch_process(self, 
                     image_paths: List[Union[str, Path]], 
                     max_workers: int = 4,
                     **kwargs) -> List[DocumentAnalysis]:
        """Process multiple documents in parallel"""
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(self.process_document, path, **kwargs)
                for path in image_paths
            ]
            
            results = []
            for future in futures:
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    logger.error(f"Error processing document: {e}")
                    results.append(None)
        
        return results
    
    def export_results(self, analysis: DocumentAnalysis, output_path: str, format: str = "json"):
        """Export analysis results to file"""
        if format.lower() == "json":
            data = {
                "full_text": analysis.full_text,
                "average_confidence": analysis.average_confidence,
                "processing_time": analysis.total_processing_time,
                "image_info": analysis.image_info,
                "preprocessing_applied": analysis.preprocessing_applied,
                "text_blocks": [
                    {
                        "text": block.text,
                        "confidence": block.confidence,
                        "bbox": block.bbox,
                        "engine": block.engine
                    }
                    for block in analysis.text_blocks
                ]
            }
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        
        elif format.lower() == "txt":
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(analysis.full_text)
        
        else:
            raise ValueError(f"Unsupported format: {format}")

# Example usage and testing
if __name__ == "__main__":
    # Initialize hybrid OCR engine
    ocr_engine = HybridOCREngine(
        primary_engine="easyocr",
        fallback_engine="tesseract",
        gpu=True
    )
    
    # Test with sample image
    # Replace with actual image path for testing
    test_image_path = "test_document.jpg"
    
    if Path(test_image_path).exists():
        # Process document
        result = ocr_engine.process_document(
            test_image_path,
            confidence_threshold=0.7,
            use_fallback=True
        )
        
        # Print results
        print(f"Extracted Text:\n{result.full_text}")
        print(f"Average Confidence: {result.average_confidence:.2f}")
        print(f"Processing Time: {result.total_processing_time:.2f}s")
        print(f"Text Blocks Found: {len(result.text_blocks)}")
        
        # Export results
        ocr_engine.export_results(result, "ocr_results.json", "json")
        ocr_engine.export_results(result, "extracted_text.txt", "txt")
        
    else:
        print("No test image found. Place a test image at 'test_document.jpg' to test the engine.")