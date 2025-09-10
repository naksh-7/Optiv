# Advanced Anonymization System with Context-Aware Masking
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from typing import List, Dict, Tuple, Optional, Union, Any
from dataclasses import dataclass
import hashlib
import secrets
import string
import re
import logging
from pathlib import Path
import json

# Import from our PII detection module
from .pii_detector import PIIEntity, PIIAnalysisResult

logger = logging.getLogger(__name__)

@dataclass
class MaskingRule:
    """Masking rule configuration"""
    entity_type: str
    strategy: str
    parameters: Dict[str, Any]
    priority: int = 1

@dataclass
class AnonymizationResult:
    """Result of anonymization operation"""
    anonymized_text: str
    anonymized_image: Optional[np.ndarray]
    mapping: Dict[str, str]  # Original -> Anonymized mapping
    entities_processed: int
    method_used: str
    success: bool
    processing_time: float

class TextAnonymizer:
    """Advanced text anonymization with multiple strategies"""
    
    def __init__(self):
        # Default masking strategies per entity type
        self.default_strategies = {
            "PERSON": "replace_realistic",
            "EMAIL_ADDRESS": "replace_domain", 
            "PHONE_NUMBER": "replace_format",
            "CREDIT_CARD": "replace_partial",
            "SSN": "replace_format",
            "IP_ADDRESS": "replace_network",
            "EMPLOYEE_ID": "replace_format",
            "ACCESS_CARD": "replace_format",
            "POLICY_NUMBER": "replace_format",
            "SYSTEM_ID": "replace_format",
            "BADGE_NUMBER": "replace_format"
        }
        
        # Realistic replacement data
        self.replacement_data = {
            "first_names": ["John", "Jane", "Michael", "Sarah", "David", "Emily", "Robert", "Lisa"],
            "last_names": ["Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller", "Davis"],
            "domains": ["example.com", "sample.org", "test.net", "demo.co"],
            "companies": ["ACME Corp", "Global Industries", "Tech Solutions", "Business Systems"]
        }
    
    def anonymize_text(self, 
                      text: str, 
                      entities: List[PIIEntity],
                      strategy: str = "auto",
                      preserve_format: bool = True) -> AnonymizationResult:
        """Anonymize text based on detected PII entities"""
        
        import time
        start_time = time.time()
        
        anonymized_text = text
        mapping = {}
        processed_entities = 0
        
        try:
            # Sort entities by start position in reverse order to avoid index shifting
            sorted_entities = sorted(entities, key=lambda x: x.start, reverse=True)
            
            for entity in sorted_entities:
                original_text = entity.text
                
                # Choose anonymization strategy
                if strategy == "auto":
                    anonymization_strategy = self.default_strategies.get(entity.label, "replace")
                else:
                    anonymization_strategy = strategy
                
                # Apply anonymization
                anonymized_value = self._apply_text_strategy(
                    original_text, 
                    entity, 
                    anonymization_strategy,
                    preserve_format
                )
                
                # Replace in text
                anonymized_text = (
                    anonymized_text[:entity.start] + 
                    anonymized_value + 
                    anonymized_text[entity.end:]
                )
                
                mapping[original_text] = anonymized_value
                processed_entities += 1
            
            return AnonymizationResult(
                anonymized_text=anonymized_text,
                anonymized_image=None,
                mapping=mapping,
                entities_processed=processed_entities,
                method_used=strategy,
                success=True,
                processing_time=time.time() - start_time
            )
            
        except Exception as e:
            logger.error(f"Error in text anonymization: {e}")
            return AnonymizationResult(
                anonymized_text=text,
                anonymized_image=None,
                mapping={},
                entities_processed=0,
                method_used=strategy,
                success=False,
                processing_time=time.time() - start_time
            )
    
    def _apply_text_strategy(self, 
                           text: str, 
                           entity: PIIEntity, 
                           strategy: str,
                           preserve_format: bool) -> str:
        """Apply specific anonymization strategy to text"""
        
        if strategy == "mask":
            return self._mask_with_asterisks(text)
        
        elif strategy == "replace":
            return f"[{entity.label}]"
        
        elif strategy == "replace_realistic":
            return self._generate_realistic_replacement(text, entity.label)
        
        elif strategy == "replace_format":
            return self._replace_preserving_format(text, entity.label, preserve_format)
        
        elif strategy == "replace_partial":
            return self._replace_partial(text)
        
        elif strategy == "replace_domain":
            return self._replace_domain(text)
        
        elif strategy == "replace_network":
            return self._replace_network_address(text)
        
        elif strategy == "hash":
            return self._hash_text(text)
        
        elif strategy == "encrypt":
            return self._encrypt_text(text)
        
        else:
            logger.warning(f"Unknown strategy: {strategy}, using replace")
            return f"[{entity.label}]"
    
    def _mask_with_asterisks(self, text: str) -> str:
        """Mask text with asterisks"""
        if len(text) <= 2:
            return "*" * len(text)
        else:
            return text[0] + "*" * (len(text) - 2) + text[-1]
    
    def _generate_realistic_replacement(self, text: str, entity_type: str) -> str:
        """Generate realistic replacement based on entity type"""
        if entity_type == "PERSON":
            first_name = secrets.choice(self.replacement_data["first_names"])
            last_name = secrets.choice(self.replacement_data["last_names"])
            return f"{first_name} {last_name}"
        
        elif entity_type == "ORGANIZATION":
            return secrets.choice(self.replacement_data["companies"])
        
        else:
            return f"[{entity_type}]"
    
    def _replace_preserving_format(self, text: str, entity_type: str, preserve_format: bool) -> str:
        """Replace while preserving original format"""
        if not preserve_format:
            return f"[{entity_type}]"
        
        # Preserve character patterns
        result = ""
        for char in text:
            if char.isdigit():
                result += str(secrets.randbelow(10))
            elif char.isalpha():
                if char.isupper():
                    result += secrets.choice(string.ascii_uppercase)
                else:
                    result += secrets.choice(string.ascii_lowercase)
            else:
                result += char
        
        return result
    
    def _replace_partial(self, text: str) -> str:
        """Replace part of the text, keeping some characters visible"""
        if len(text) <= 4:
            return "*" * len(text)
        
        # Show first 2 and last 2 characters
        visible_chars = 4
        middle_length = len(text) - visible_chars
        
        return text[:2] + "*" * middle_length + text[-2:]
    
    def _replace_domain(self, text: str) -> str:
        """Replace email domain while keeping local part structure"""
        if "@" in text:
            local, domain = text.split("@", 1)
            # Keep local part structure but anonymize
            anonymized_local = re.sub(r'[a-zA-Z]', 'x', local)
            anonymized_local = re.sub(r'[0-9]', '0', anonymized_local)
            new_domain = secrets.choice(self.replacement_data["domains"])
            return f"{anonymized_local}@{new_domain}"
        
        return text
    
    def _replace_network_address(self, text: str) -> str:
        """Replace IP address with similar format"""
        # Replace IP with random private IP
        return f"192.168.{secrets.randbelow(255)}.{secrets.randbelow(255)}"
    
    def _hash_text(self, text: str) -> str:
        """Hash text using SHA-256"""
        return hashlib.sha256(text.encode()).hexdigest()[:16]
    
    def _encrypt_text(self, text: str) -> str:
        """Simple encryption (for demo purposes)"""
        # In production, use proper encryption
        encoded = ""
        for char in text:
            encoded += f"{ord(char):02x}"
        return f"ENC_{encoded}"

class ImageAnonymizer:
    """Image-based anonymization for visual PII"""
    
    def __init__(self):
        self.blur_kernel = (51, 51)
        self.pixelate_factor = 20
    
    def anonymize_image_regions(self, 
                               image: np.ndarray,
                               regions: List[Tuple[int, int, int, int]],
                               method: str = "blur") -> np.ndarray:
        """Anonymize specific regions in image"""
        
        anonymized = image.copy()
        
        for x1, y1, x2, y2 in regions:
            # Ensure coordinates are within image bounds
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(image.shape[1], x2)
            y2 = min(image.shape[0], y2)
            
            if method == "blur":
                anonymized = self._blur_region(anonymized, (x1, y1, x2, y2))
            elif method == "pixelate":
                anonymized = self._pixelate_region(anonymized, (x1, y1, x2, y2))
            elif method == "black_box":
                anonymized = self._black_box_region(anonymized, (x1, y1, x2, y2))
            elif method == "white_box":
                anonymized = self._white_box_region(anonymized, (x1, y1, x2, y2))
            else:
                logger.warning(f"Unknown image anonymization method: {method}")
                anonymized = self._blur_region(anonymized, (x1, y1, x2, y2))
        
        return anonymized
    
    def _blur_region(self, image: np.ndarray, region: Tuple[int, int, int, int]) -> np.ndarray:
        """Apply Gaussian blur to region"""
        x1, y1, x2, y2 = region
        roi = image[y1:y2, x1:x2]
        
        if roi.size > 0:
            blurred_roi = cv2.GaussianBlur(roi, self.blur_kernel, 0)
            image[y1:y2, x1:x2] = blurred_roi
        
        return image
    
    def _pixelate_region(self, image: np.ndarray, region: Tuple[int, int, int, int]) -> np.ndarray:
        """Apply pixelation to region"""
        x1, y1, x2, y2 = region
        roi = image[y1:y2, x1:x2]
        
        if roi.size > 0:
            h, w = roi.shape[:2]
            # Downsample
            small = cv2.resize(roi, (w // self.pixelate_factor, h // self.pixelate_factor))
            # Upsample back
            pixelated = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)
            image[y1:y2, x1:x2] = pixelated
        
        return image
    
    def _black_box_region(self, image: np.ndarray, region: Tuple[int, int, int, int]) -> np.ndarray:
        """Cover region with black rectangle"""
        x1, y1, x2, y2 = region
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 0), -1)
        return image
    
    def _white_box_region(self, image: np.ndarray, region: Tuple[int, int, int, int]) -> np.ndarray:
        """Cover region with white rectangle"""
        x1, y1, x2, y2 = region
        cv2.rectangle(image, (x1, y1), (x2, y2), (255, 255, 255), -1)
        return image

class AdvancedAnonymizer:
    """Advanced anonymization system combining text and image processing"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.text_anonymizer = TextAnonymizer()
        self.image_anonymizer = ImageAnonymizer()
        
        # Default configuration
        self.config = config or {
            "preserve_format": True,
            "use_realistic_replacements": True,
            "image_anonymization_method": "blur",
            "maintain_audit_trail": True
        }
        
        # Audit trail for compliance
        self.audit_trail = []
    
    def anonymize_document(self, 
                          text: str,
                          pii_analysis: PIIAnalysisResult,
                          image: Optional[np.ndarray] = None,
                          image_regions: Optional[List[Tuple[int, int, int, int]]] = None,
                          custom_rules: Optional[List[MaskingRule]] = None) -> AnonymizationResult:
        """Anonymize complete document with text and image"""
        
        import time
        start_time = time.time()
        
        try:
            # Apply custom rules if provided
            entities_to_process = self._apply_custom_rules(pii_analysis.entities, custom_rules)
            
            # Anonymize text
            text_result = self.text_anonymizer.anonymize_text(
                text, 
                entities_to_process,
                strategy="auto",
                preserve_format=self.config["preserve_format"]
            )
            
            # Anonymize image if provided
            anonymized_image = None
            if image is not None and image_regions:
                anonymized_image = self.image_anonymizer.anonymize_image_regions(
                    image,
                    image_regions,
                    method=self.config["image_anonymization_method"]
                )
            
            # Create audit trail entry
            if self.config["maintain_audit_trail"]:
                audit_entry = {
                    "timestamp": time.time(),
                    "entities_processed": len(entities_to_process),
                    "text_anonymized": len(text_result.anonymized_text) > 0,
                    "image_anonymized": anonymized_image is not None,
                    "method": "advanced_anonymization"
                }
                self.audit_trail.append(audit_entry)
            
            return AnonymizationResult(
                anonymized_text=text_result.anonymized_text,
                anonymized_image=anonymized_image,
                mapping=text_result.mapping,
                entities_processed=text_result.entities_processed,
                method_used="advanced",
                success=True,
                processing_time=time.time() - start_time
            )
            
        except Exception as e:
            logger.error(f"Error in document anonymization: {e}")
            return AnonymizationResult(
                anonymized_text=text,
                anonymized_image=image,
                mapping={},
                entities_processed=0,
                method_used="advanced",
                success=False,
                processing_time=time.time() - start_time
            )
    
    def _apply_custom_rules(self, 
                           entities: List[PIIEntity], 
                           custom_rules: Optional[List[MaskingRule]]) -> List[PIIEntity]:
        """Apply custom masking rules to filter/modify entities"""
        
        if not custom_rules:
            return entities
        
        # Create rule lookup
        rule_lookup = {rule.entity_type: rule for rule in custom_rules}
        
        filtered_entities = []
        
        for entity in entities:
            # Check if there's a custom rule for this entity type
            if entity.label in rule_lookup:
                rule = rule_lookup[entity.label]
                
                # Apply rule based on strategy
                if rule.strategy == "skip":
                    continue  # Skip this entity
                
                elif rule.strategy == "lower_threshold":
                    min_confidence = rule.parameters.get("min_confidence", 0.9)
                    if entity.confidence < min_confidence:
                        continue
                
                elif rule.strategy == "whitelist":
                    whitelist = rule.parameters.get("allowed_values", [])
                    if entity.text in whitelist:
                        continue
                
                elif rule.strategy == "blacklist":
                    blacklist = rule.parameters.get("blocked_values", [])
                    if entity.text in blacklist:
                        # Force anonymization
                        entity.confidence = 1.0
            
            filtered_entities.append(entity)
        
        return filtered_entities
    
    def create_anonymization_report(self, 
                                   results: List[AnonymizationResult]) -> Dict[str, Any]:
        """Create comprehensive anonymization report"""
        
        if not results:
            return {"error": "No results provided"}
        
        total_entities = sum(r.entities_processed for r in results)
        successful_operations = len([r for r in results if r.success])
        total_processing_time = sum(r.processing_time for r in results)
        
        # Entity type analysis
        all_mappings = {}
        for result in results:
            all_mappings.update(result.mapping)
        
        report = {
            "summary": {
                "total_documents_processed": len(results),
                "successful_operations": successful_operations,
                "total_entities_anonymized": total_entities,
                "total_processing_time": total_processing_time,
                "average_processing_time": total_processing_time / len(results) if results else 0
            },
            "entity_analysis": {
                "total_unique_values_anonymized": len(all_mappings),
                "anonymization_methods_used": list(set([r.method_used for r in results]))
            },
            "audit_trail": self.audit_trail[-10:],  # Last 10 entries
            "compliance_info": {
                "anonymization_performed": total_entities > 0,
                "audit_trail_maintained": self.config["maintain_audit_trail"],
                "format_preservation": self.config["preserve_format"]
            }
        }
        
        return report
    
    def export_anonymization_mapping(self, 
                                   results: List[AnonymizationResult],
                                   output_path: str,
                                   include_hashes: bool = True) -> None:
        """Export anonymization mapping for audit purposes"""
        
        combined_mapping = {}
        for result in results:
            combined_mapping.update(result.mapping)
        
        export_data = {
            "timestamp": time.time(),
            "total_mappings": len(combined_mapping),
            "mappings": {}
        }
        
        for original, anonymized in combined_mapping.items():
            mapping_entry = {
                "anonymized_value": anonymized,
                "length_original": len(original),
                "length_anonymized": len(anonymized)
            }
            
            if include_hashes:
                mapping_entry["original_hash"] = hashlib.sha256(original.encode()).hexdigest()
            
            export_data["mappings"][original] = mapping_entry
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)
    
    def reverse_anonymization(self, 
                             anonymized_text: str,
                             mapping: Dict[str, str]) -> str:
        """Reverse anonymization using provided mapping (for authorized personnel)"""
        
        # Create reverse mapping
        reverse_mapping = {v: k for k, v in mapping.items()}
        
        result_text = anonymized_text
        
        # Sort by length (longest first) to avoid partial replacements
        sorted_items = sorted(reverse_mapping.items(), key=lambda x: len(x[0]), reverse=True)
        
        for anonymized, original in sorted_items:
            result_text = result_text.replace(anonymized, original)
        
        return result_text

# Example usage and testing
if __name__ == "__main__":
    # Example usage of the anonymization system
    
    # Sample text with PII
    sample_text = """
    Employee John Smith (ID: EMP123456) can be reached at john.smith@company.com.
    His phone number is (555) 123-4567 and SSN is 123-45-6789.
    He accessed server SYS-PROD-01 using badge B-445566.
    """
    
    # Mock PII entities (normally from PII detection)
    mock_entities = [
        PIIEntity("John Smith", "PERSON", 9, 19, 0.95, "spacy"),
        PIIEntity("EMP123456", "EMPLOYEE_ID", 25, 34, 0.90, "custom_patterns"),
        PIIEntity("john.smith@company.com", "EMAIL_ADDRESS", 54, 77, 0.98, "presidio"),
        PIIEntity("(555) 123-4567", "PHONE_NUMBER", 101, 115, 0.92, "presidio"),
        PIIEntity("123-45-6789", "SSN", 127, 138, 0.99, "presidio"),
        PIIEntity("SYS-PROD-01", "SYSTEM_ID", 157, 168, 0.88, "custom_patterns"),
        PIIEntity("B-445566", "BADGE_NUMBER", 181, 189, 0.85, "custom_patterns")
    ]
    
    # Mock PII analysis result
    from .pii_detector import PIIAnalysisResult
    mock_analysis = PIIAnalysisResult(
        entities=mock_entities,
        text=sample_text,
        confidence_scores={"presidio": 0.95, "spacy": 0.90, "custom_patterns": 0.88},
        processing_time=0.5,
        engines_used=["presidio", "spacy", "custom_patterns"],
        context_analysis={}
    )
    
    # Initialize anonymizer
    anonymizer = AdvancedAnonymizer({
        "preserve_format": True,
        "use_realistic_replacements": True,
        "maintain_audit_trail": True
    })
    
    # Anonymize document
    result = anonymizer.anonymize_document(
        sample_text,
        mock_analysis
    )
    
    # Print results
    print("Original text:")
    print(sample_text)
    print("\n" + "="*50)
    print("Anonymized text:")
    print(result.anonymized_text)
    print("\n" + "="*50)
    print(f"Entities processed: {result.entities_processed}")
    print(f"Processing time: {result.processing_time:.2f}s")
    print(f"Success: {result.success}")
    
    print("\nMapping:")
    for original, anonymized in result.mapping.items():
        print(f"'{original}' -> '{anonymized}'")
    
    # Generate report
    report = anonymizer.create_anonymization_report([result])
    print(f"\nAnonymization Report:")
    print(json.dumps(report, indent=2))