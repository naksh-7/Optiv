# Main Document Processing Orchestrator
import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional, Union, Any
from dataclasses import dataclass
from pathlib import Path
import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

# Import our core modules
from .ocr_engine import HybridOCREngine, DocumentAnalysis, OCRResult
from .pii_detector import MultiEnginePIIDetector, PIIAnalysisResult, PIIEntity
from .logo_detector import LogoDetectionPipeline, LogoDetection
from .anonymizer import AdvancedAnonymizer, AnonymizationResult

logger = logging.getLogger(__name__)

@dataclass
class ProcessingMetrics:
    """Processing performance metrics"""
    total_time: float
    ocr_time: float
    pii_detection_time: float
    logo_detection_time: float
    anonymization_time: float
    confidence_scores: Dict[str, float]
    entities_found: int
    logos_found: int

@dataclass
class DocumentProcessingResult:
    """Complete document processing result"""
    # Input information
    document_id: str
    original_image: Optional[np.ndarray]
    
    # OCR results
    ocr_analysis: DocumentAnalysis
    extracted_text: str
    
    # PII detection results
    pii_analysis: PIIAnalysisResult
    detected_entities: List[PIIEntity]
    
    # Logo detection results
    logo_detections: List[LogoDetection]
    logo_removed_image: Optional[np.ndarray]
    
    # Anonymization results
    anonymization_result: AnonymizationResult
    final_text: str
    final_image: Optional[np.ndarray]
    
    # Metadata
    processing_metrics: ProcessingMetrics
    success: bool
    error_message: Optional[str] = None

class DocumentProcessor:
    """Main orchestrator for complete document processing pipeline"""
    
    def __init__(self, config: Dict[str, Any] = None):
        # Default configuration
        self.config = config or {
            "ocr": {
                "primary_engine": "easyocr",
                "fallback_engine": "tesseract",
                "confidence_threshold": 0.7,
                "use_preprocessing": True
            },
            "pii_detection": {
                "engines": ["presidio", "spacy", "custom_patterns"],
                "confidence_threshold": 0.8,
                "enable_context_analysis": True
            },
            "logo_detection": {
                "confidence_threshold": 0.6,
                "removal_method": "inpaint",
                "selective_removal": True
            },
            "anonymization": {
                "strategy": "auto",
                "preserve_format": True,
                "use_realistic_replacements": True
            },
            "processing": {
                "parallel_processing": True,
                "max_workers": 4,
                "timeout_seconds": 300
            }
        }
        
        # Initialize processing engines
        self._initialize_engines()
    
    def _initialize_engines(self):
        """Initialize all processing engines"""
        try:
            # OCR Engine
            self.ocr_engine = HybridOCREngine(
                primary_engine=self.config["ocr"]["primary_engine"],
                fallback_engine=self.config["ocr"]["fallback_engine"],
                gpu=True
            )
            
            # PII Detection Engine
            self.pii_detector = MultiEnginePIIDetector(
                engines=self.config["pii_detection"]["engines"],
                enable_context_analysis=self.config["pii_detection"]["enable_context_analysis"],
                confidence_threshold=self.config["pii_detection"]["confidence_threshold"]
            )
            
            # Logo Detection Pipeline
            self.logo_pipeline = LogoDetectionPipeline({
                "confidence_threshold": self.config["logo_detection"]["confidence_threshold"],
                "device": "auto"
            })
            
            # Anonymization System
            self.anonymizer = AdvancedAnonymizer({
                "preserve_format": self.config["anonymization"]["preserve_format"],
                "use_realistic_replacements": self.config["anonymization"]["use_realistic_replacements"],
                "maintain_audit_trail": True
            })
            
            logger.info("All processing engines initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing processing engines: {e}")
            raise
    
    def process_document(self, 
                        document: Union[str, Path, np.ndarray],
                        document_id: Optional[str] = None,
                        processing_options: Optional[Dict[str, Any]] = None) -> DocumentProcessingResult:
        """Process a complete document through the entire pipeline"""
        
        start_time = time.time()
        
        # Generate document ID if not provided
        if document_id is None:
            document_id = f"doc_{int(time.time() * 1000)}"
        
        # Merge processing options with config
        options = self.config.copy()
        if processing_options:
            self._merge_config(options, processing_options)
        
        logger.info(f"Starting processing for document: {document_id}")
        
        try:
            # Load image
            if isinstance(document, (str, Path)):
                original_image = cv2.imread(str(document))
                if original_image is None:
                    raise ValueError(f"Could not load image from {document}")
            else:
                original_image = document.copy()
            
            # Step 1: OCR Processing
            ocr_start = time.time()
            logger.info("Step 1: OCR text extraction")
            
            ocr_analysis = self.ocr_engine.process_document(
                original_image,
                confidence_threshold=options["ocr"]["confidence_threshold"],
                use_fallback=True
            )
            
            extracted_text = ocr_analysis.full_text
            ocr_time = time.time() - ocr_start
            
            logger.info(f"OCR completed in {ocr_time:.2f}s, extracted {len(extracted_text)} characters")
            
            # Step 2: PII Detection
            pii_start = time.time()
            logger.info("Step 2: PII detection")
            
            pii_analysis = self.pii_detector.detect_pii(extracted_text)
            detected_entities = pii_analysis.entities
            pii_time = time.time() - pii_start
            
            logger.info(f"PII detection completed in {pii_time:.2f}s, found {len(detected_entities)} entities")
            
            # Step 3: Logo Detection and Removal
            logo_start = time.time()
            logger.info("Step 3: Logo detection and removal")
            
            logo_results = self.logo_pipeline.process_document(
                original_image,
                remove_logos=True,
                removal_method=options["logo_detection"]["removal_method"]
            )
            
            logo_detections = logo_results.get("detections", [])
            logo_removed_image = logo_results.get("processed_image", original_image)
            logo_time = time.time() - logo_start
            
            logger.info(f"Logo processing completed in {logo_time:.2f}s, found {len(logo_detections)} logos")
            
            # Step 4: Anonymization
            anon_start = time.time()
            logger.info("Step 4: Text and image anonymization")
            
            # Get regions from OCR results for image anonymization
            text_regions = [(result.bbox[0], result.bbox[1], result.bbox[2], result.bbox[3]) 
                           for result in ocr_analysis.text_blocks]
            
            anonymization_result = self.anonymizer.anonymize_document(
                extracted_text,
                pii_analysis,
                image=logo_removed_image,
                image_regions=text_regions if detected_entities else None
            )
            
            final_text = anonymization_result.anonymized_text
            final_image = anonymization_result.anonymized_image or logo_removed_image
            anon_time = time.time() - anon_start
            
            logger.info(f"Anonymization completed in {anon_time:.2f}s")
            
            # Calculate metrics
            total_time = time.time() - start_time
            metrics = ProcessingMetrics(
                total_time=total_time,
                ocr_time=ocr_time,
                pii_detection_time=pii_time,
                logo_detection_time=logo_time,
                anonymization_time=anon_time,
                confidence_scores={
                    "ocr_confidence": ocr_analysis.average_confidence,
                    "pii_confidence": np.mean([e.confidence for e in detected_entities]) if detected_entities else 0.0,
                    "logo_confidence": np.mean([d.confidence for d in logo_detections]) if logo_detections else 0.0
                },
                entities_found=len(detected_entities),
                logos_found=len(logo_detections)
            )
            
            # Create result
            result = DocumentProcessingResult(
                document_id=document_id,
                original_image=original_image,
                ocr_analysis=ocr_analysis,
                extracted_text=extracted_text,
                pii_analysis=pii_analysis,
                detected_entities=detected_entities,
                logo_detections=logo_detections,
                logo_removed_image=logo_removed_image,
                anonymization_result=anonymization_result,
                final_text=final_text,
                final_image=final_image,
                processing_metrics=metrics,
                success=True
            )
            
            logger.info(f"Document {document_id} processed successfully in {total_time:.2f}s")
            return result
            
        except Exception as e:
            error_msg = f"Error processing document {document_id}: {e}"
            logger.error(error_msg)
            
            # Return error result
            return DocumentProcessingResult(
                document_id=document_id,
                original_image=original_image if 'original_image' in locals() else None,
                ocr_analysis=ocr_analysis if 'ocr_analysis' in locals() else None,
                extracted_text="",
                pii_analysis=None,
                detected_entities=[],
                logo_detections=[],
                logo_removed_image=None,
                anonymization_result=None,
                final_text="",
                final_image=None,
                processing_metrics=ProcessingMetrics(
                    total_time=time.time() - start_time,
                    ocr_time=0, pii_detection_time=0, logo_detection_time=0, anonymization_time=0,
                    confidence_scores={}, entities_found=0, logos_found=0
                ),
                success=False,
                error_message=error_msg
            )
    
    def batch_process_documents(self, 
                               documents: List[Union[str, Path, np.ndarray]],
                               output_directory: Optional[str] = None) -> List[DocumentProcessingResult]:
        """Process multiple documents in parallel"""
        
        if not documents:
            return []
        
        logger.info(f"Starting batch processing of {len(documents)} documents")
        
        results = []
        
        if self.config["processing"]["parallel_processing"]:
            # Parallel processing
            max_workers = min(self.config["processing"]["max_workers"], len(documents))
            
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                # Submit all tasks
                future_to_doc = {
                    executor.submit(
                        self.process_document, 
                        doc, 
                        f"batch_doc_{i}"
                    ): (i, doc) 
                    for i, doc in enumerate(documents)
                }
                
                # Collect results
                for future in as_completed(future_to_doc, 
                                         timeout=self.config["processing"]["timeout_seconds"]):
                    doc_index, doc = future_to_doc[future]
                    try:
                        result = future.result()
                        results.append((doc_index, result))
                    except Exception as e:
                        logger.error(f"Error processing document {doc_index}: {e}")
                        # Create error result
                        error_result = DocumentProcessingResult(
                            document_id=f"batch_doc_{doc_index}",
                            original_image=None, ocr_analysis=None, extracted_text="",
                            pii_analysis=None, detected_entities=[], logo_detections=[],
                            logo_removed_image=None, anonymization_result=None,
                            final_text="", final_image=None,
                            processing_metrics=ProcessingMetrics(
                                0, 0, 0, 0, 0, {}, 0, 0
                            ),
                            success=False, error_message=str(e)
                        )
                        results.append((doc_index, error_result))
        
        else:
            # Sequential processing
            for i, doc in enumerate(documents):
                result = self.process_document(doc, f"batch_doc_{i}")
                results.append((i, result))
        
        # Sort results by original order
        results.sort(key=lambda x: x[0])
        final_results = [result for _, result in results]
        
        # Save results if output directory specified
        if output_directory:
            self.save_batch_results(final_results, output_directory)
        
        logger.info(f"Batch processing completed. {len(final_results)} documents processed")
        return final_results
    
    def save_batch_results(self, 
                          results: List[DocumentProcessingResult], 
                          output_directory: str):
        """Save batch processing results to directory"""
        
        output_path = Path(output_directory)
        output_path.mkdir(parents=True, exist_ok=True)
        
        successful_results = [r for r in results if r.success]
        failed_results = [r for r in results if not r.success]
        
        # Save summary
        summary = {
            "total_documents": len(results),
            "successful": len(successful_results),
            "failed": len(failed_results),
            "average_processing_time": np.mean([r.processing_metrics.total_time for r in successful_results]) if successful_results else 0,
            "total_entities_found": sum([r.processing_metrics.entities_found for r in successful_results]),
            "total_logos_found": sum([r.processing_metrics.logos_found for r in successful_results])
        }
        
        with open(output_path / "batch_summary.json", 'w') as f:
            json.dump(summary, f, indent=2)
        
        # Save individual results
        for result in results:
            doc_dir = output_path / result.document_id
            doc_dir.mkdir(exist_ok=True)
            
            if result.success:
                # Save final text
                with open(doc_dir / "anonymized_text.txt", 'w', encoding='utf-8') as f:
                    f.write(result.final_text)
                
                # Save final image
                if result.final_image is not None:
                    cv2.imwrite(str(doc_dir / "anonymized_image.jpg"), result.final_image)
                
                # Save detailed results
                result_data = {
                    "document_id": result.document_id,
                    "extracted_text_length": len(result.extracted_text),
                    "entities_found": len(result.detected_entities),
                    "logos_found": len(result.logo_detections),
                    "processing_metrics": {
                        "total_time": result.processing_metrics.total_time,
                        "confidence_scores": result.processing_metrics.confidence_scores
                    },
                    "anonymization_mapping": result.anonymization_result.mapping if result.anonymization_result else {}
                }
                
                with open(doc_dir / "processing_results.json", 'w') as f:
                    json.dump(result_data, f, indent=2)
            
            else:
                # Save error information
                with open(doc_dir / "error.txt", 'w') as f:
                    f.write(result.error_message or "Unknown error")
        
        logger.info(f"Batch results saved to {output_directory}")
    
    def _merge_config(self, base_config: Dict[str, Any], updates: Dict[str, Any]):
        """Recursively merge configuration updates"""
        for key, value in updates.items():
            if key in base_config and isinstance(base_config[key], dict) and isinstance(value, dict):
                self._merge_config(base_config[key], value)
            else:
                base_config[key] = value
    
    def get_processing_statistics(self, 
                                results: List[DocumentProcessingResult]) -> Dict[str, Any]:
        """Generate comprehensive processing statistics"""
        
        if not results:
            return {}
        
        successful_results = [r for r in results if r.success]
        
        if not successful_results:
            return {"error": "No successful processing results"}
        
        # Time statistics
        processing_times = [r.processing_metrics.total_time for r in successful_results]
        ocr_times = [r.processing_metrics.ocr_time for r in successful_results]
        pii_times = [r.processing_metrics.pii_detection_time for r in successful_results]
        logo_times = [r.processing_metrics.logo_detection_time for r in successful_results]
        anon_times = [r.processing_metrics.anonymization_time for r in successful_results]
        
        # Entity statistics
        all_entities = []
        for result in successful_results:
            all_entities.extend(result.detected_entities)
        
        entity_types = {}
        for entity in all_entities:
            entity_types[entity.label] = entity_types.get(entity.label, 0) + 1
        
        # Confidence statistics
        ocr_confidences = [r.processing_metrics.confidence_scores.get("ocr_confidence", 0) for r in successful_results]
        pii_confidences = [r.processing_metrics.confidence_scores.get("pii_confidence", 0) for r in successful_results]
        
        return {
            "processing_summary": {
                "total_documents": len(results),
                "successful_documents": len(successful_results),
                "success_rate": len(successful_results) / len(results) * 100,
                "total_entities_found": len(all_entities),
                "total_logos_found": sum([r.processing_metrics.logos_found for r in successful_results])
            },
            "performance_metrics": {
                "average_total_time": np.mean(processing_times),
                "average_ocr_time": np.mean(ocr_times),
                "average_pii_time": np.mean(pii_times),
                "average_logo_time": np.mean(logo_times),
                "average_anonymization_time": np.mean(anon_times),
                "throughput_docs_per_second": len(successful_results) / sum(processing_times) if sum(processing_times) > 0 else 0
            },
            "entity_analysis": {
                "entity_type_distribution": entity_types,
                "average_entities_per_document": len(all_entities) / len(successful_results),
                "most_common_entity_type": max(entity_types, key=entity_types.get) if entity_types else None
            },
            "confidence_analysis": {
                "average_ocr_confidence": np.mean(ocr_confidences),
                "average_pii_confidence": np.mean(pii_confidences),
                "ocr_confidence_std": np.std(ocr_confidences),
                "pii_confidence_std": np.std(pii_confidences)
            }
        }

# Example usage
if __name__ == "__main__":
    # Initialize document processor
    processor = DocumentProcessor({
        "ocr": {"confidence_threshold": 0.7},
        "pii_detection": {"confidence_threshold": 0.8},
        "logo_detection": {"confidence_threshold": 0.6},
        "processing": {"parallel_processing": True, "max_workers": 2}
    })
    
    # Test with single document
    test_image = "test_document.jpg"
    
    if Path(test_image).exists():
        result = processor.process_document(test_image)
        
        if result.success:
            print(f"Document processed successfully!")
            print(f"Extracted text length: {len(result.extracted_text)}")
            print(f"Entities found: {len(result.detected_entities)}")
            print(f"Logos found: {len(result.logo_detections)}")
            print(f"Processing time: {result.processing_metrics.total_time:.2f}s")
        else:
            print(f"Processing failed: {result.error_message}")
    
    else:
        print("Test image not found. Place test_document.jpg to test the system.")