# Multi-Engine PII Detection System
import spacy
import re
from typing import List, Dict, Tuple, Optional, Union, Any
from dataclasses import dataclass
import numpy as np
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
from presidio_analyzer import AnalyzerEngine, Pattern, PatternRecognizer
from presidio_analyzer.nlp_engine import NlpEngineProvider
import logging
from concurrent.futures import ThreadPoolExecutor
import time

logger = logging.getLogger(__name__)

@dataclass
class PIIEntity:
    """PII entity detection result"""
    text: str
    label: str
    start: int
    end: int
    confidence: float
    engine: str
    context_score: float = 0.0
    is_sensitive: bool = True

@dataclass
class PIIAnalysisResult:
    """Complete PII analysis result"""
    entities: List[PIIEntity]
    text: str
    confidence_scores: Dict[str, float]
    processing_time: float
    engines_used: List[str]
    context_analysis: Dict[str, Any]

class SecurityPatternRecognizer:
    """Custom pattern recognizer for security-specific entities"""
    
    def __init__(self):
        self.patterns = {
            "EMPLOYEE_ID": [
                r"\b(EMP|EMPLOYEE)[-_]?\d{4,8}\b",
                r"\bEMP\d{6}\b",
                r"\b[A-Z]{2,3}[-_]?\d{4,6}\b"
            ],
            "ACCESS_CARD": [
                r"\b(AC|ACCESS)[-_]?\d{4,8}\b",
                r"\bCARD[-_]?\d{4,8}\b",
                r"\b[A-Z]{2}[-_]\d{6}\b"
            ],
            "POLICY_NUMBER": [
                r"\b(POL|POLICY)[-_][A-Z0-9-]{6,15}\b",
                r"\bP\d{4}[-_]\d{3}\b",
                r"\bPOLICY[-_]\d{4}[-_]\d{3}\b"
            ],
            "SYSTEM_ID": [
                r"\b(SYS|SYSTEM)[-_][A-Z0-9-]{6,15}\b",
                r"\bSRV[-_]\d{3,6}\b",
                r"\bSERVER[-_][A-Z0-9]{4,8}\b"
            ],
            "BADGE_NUMBER": [
                r"\b(B|BADGE)[-_]?\d{4,8}\b",
                r"\bBDG\d{6}\b"
            ],
            "AUDIT_REF": [
                r"\b(AUD|AUDIT)[-_]\d{4}[-_][A-Z0-9-]{6,10}\b",
                r"\bAUDIT[-_]REF[-_][A-Z0-9]+\b"
            ],
            "CREDENTIAL_ID": [
                r"\b(CRED|CREDENTIAL)[-_]\d{6}[-_][A-Z]\b",
                r"\bCRED\d{8}\b"
            ],
            "ROOM_NUMBER": [
                r"\b(ROOM|RM)[-_]?\d{3}[A-Z]?\b",
                r"\b\d{3}[A-Z]?(?=\s*(room|office|door))\b"
            ],
            "IP_ADDRESS": [
                r"\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b"
            ],
            "MAC_ADDRESS": [
                r"\b([0-9A-Fa-f]{2}[:-]){5}([0-9A-Fa-f]{2})\b"
            ]
        }
    
    def detect_patterns(self, text: str, confidence_base: float = 0.8) -> List[PIIEntity]:
        """Detect custom security patterns"""
        entities = []
        
        for entity_type, patterns in self.patterns.items():
            for pattern in patterns:
                matches = re.finditer(pattern, text, re.IGNORECASE)
                
                for match in matches:
                    # Calculate confidence based on pattern specificity
                    confidence = confidence_base
                    
                    # Boost confidence for more specific patterns
                    if len(match.group()) > 8:
                        confidence += 0.1
                    if '-' in match.group() or '_' in match.group():
                        confidence += 0.05
                    
                    confidence = min(confidence, 1.0)
                    
                    entities.append(PIIEntity(
                        text=match.group(),
                        label=entity_type,
                        start=match.start(),
                        end=match.end(),
                        confidence=confidence,
                        engine="custom_patterns",
                        is_sensitive=True
                    ))
        
        return entities

class SpacyPIIDetector:
    """spaCy-based PII detection"""
    
    def __init__(self, model_name: str = "en_core_web_lg"):
        try:
            self.nlp = spacy.load(model_name)
        except OSError:
            logger.warning(f"Model {model_name} not found, using en_core_web_sm")
            self.nlp = spacy.load("en_core_web_sm")
        
        # Entity mappings to PII types
        self.entity_mappings = {
            "PERSON": "PERSON",
            "ORG": "ORGANIZATION", 
            "GPE": "LOCATION",
            "DATE": "DATE",
            "TIME": "TIME",
            "MONEY": "FINANCIAL",
            "CARDINAL": "NUMBER",
            "ORDINAL": "NUMBER"
        }
    
    def detect_entities(self, text: str, confidence_threshold: float = 0.7) -> List[PIIEntity]:
        """Detect PII entities using spaCy"""
        doc = self.nlp(text)
        entities = []
        
        for ent in doc.ents:
            if ent.label_ in self.entity_mappings:
                # spaCy doesn't provide confidence scores by default
                # Use heuristic based on entity properties
                confidence = self._calculate_confidence(ent, doc)
                
                if confidence >= confidence_threshold:
                    entities.append(PIIEntity(
                        text=ent.text,
                        label=self.entity_mappings[ent.label_],
                        start=ent.start_char,
                        end=ent.end_char,
                        confidence=confidence,
                        engine="spacy",
                        is_sensitive=self._is_sensitive_entity(ent.label_)
                    ))
        
        return entities
    
    def _calculate_confidence(self, entity, doc) -> float:
        """Calculate confidence score heuristically"""
        base_confidence = 0.8
        
        # Boost confidence for longer entities
        if len(entity.text) > 10:
            base_confidence += 0.1
        
        # Boost for entities with proper capitalization
        if entity.text.istitle():
            base_confidence += 0.05
        
        # Reduce for very short entities
        if len(entity.text) <= 2:
            base_confidence -= 0.2
        
        return min(max(base_confidence, 0.0), 1.0)
    
    def _is_sensitive_entity(self, label: str) -> bool:
        """Determine if entity type is sensitive"""
        sensitive_types = {"PERSON", "ORG", "GPE", "MONEY"}
        return label in sensitive_types

class TransformerPIIDetector:
    """Transformer-based PII detection using Hugging Face models"""
    
    def __init__(self, model_name: str = "dbmdz/bert-large-cased-finetuned-conll03-english"):
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForTokenClassification.from_pretrained(model_name)
            self.pipeline = pipeline(
                "ner",
                model=self.model,
                tokenizer=self.tokenizer,
                aggregation_strategy="simple"
            )
        except Exception as e:
            logger.error(f"Error loading transformer model: {e}")
            self.pipeline = None
    
    def detect_entities(self, text: str, confidence_threshold: float = 0.8) -> List[PIIEntity]:
        """Detect PII entities using transformer model"""
        if self.pipeline is None:
            return []
        
        try:
            results = self.pipeline(text)
            entities = []
            
            for result in results:
                if result['score'] >= confidence_threshold:
                    entities.append(PIIEntity(
                        text=result['word'],
                        label=self._map_label(result['entity_group']),
                        start=result['start'],
                        end=result['end'],
                        confidence=result['score'],
                        engine="transformer",
                        is_sensitive=self._is_sensitive_label(result['entity_group'])
                    ))
            
            return entities
            
        except Exception as e:
            logger.error(f"Error in transformer PII detection: {e}")
            return []
    
    def _map_label(self, label: str) -> str:
        """Map transformer labels to standard PII labels"""
        mappings = {
            "PER": "PERSON",
            "PERSON": "PERSON", 
            "ORG": "ORGANIZATION",
            "LOC": "LOCATION",
            "MISC": "MISCELLANEOUS"
        }
        return mappings.get(label.upper(), label)
    
    def _is_sensitive_label(self, label: str) -> bool:
        """Check if label represents sensitive information"""
        sensitive = {"PER", "PERSON", "ORG", "ORGANIZATION"}
        return label.upper() in sensitive

class PresidioPIIDetector:
    """Microsoft Presidio PII detection wrapper"""
    
    def __init__(self):
        try:
            # Initialize Presidio analyzer
            self.analyzer = AnalyzerEngine()
            
            # Add custom recognizers
            self._add_custom_recognizers()
            
        except Exception as e:
            logger.error(f"Error initializing Presidio: {e}")
            self.analyzer = None
    
    def _add_custom_recognizers(self):
        """Add custom recognizers to Presidio"""
        # Security-specific patterns
        security_patterns = [
            Pattern("Employee ID", r"EMP\d{6}", 0.8),
            Pattern("Access Card", r"AC-\d{6}", 0.8),
            Pattern("Policy Number", r"POL-[A-Z0-9-]+", 0.8),
            Pattern("Badge Number", r"B-\d{6}", 0.8),
            Pattern("System ID", r"SYS-[A-Z0-9-]+", 0.8),
        ]
        
        security_recognizer = PatternRecognizer(
            supported_entity="SECURITY_ID",
            patterns=security_patterns,
            context=["employee", "access", "card", "badge", "policy", "system"]
        )
        
        self.analyzer.registry.add_recognizer(security_recognizer)
    
    def detect_entities(self, text: str, language: str = "en") -> List[PIIEntity]:
        """Detect PII entities using Presidio"""
        if self.analyzer is None:
            return []
        
        try:
            results = self.analyzer.analyze(text=text, language=language)
            entities = []
            
            for result in results:
                entities.append(PIIEntity(
                    text=text[result.start:result.end],
                    label=result.entity_type,
                    start=result.start,
                    end=result.end,
                    confidence=result.score,
                    engine="presidio",
                    is_sensitive=True
                ))
            
            return entities
            
        except Exception as e:
            logger.error(f"Error in Presidio PII detection: {e}")
            return []

class ContextAnalyzer:
    """Analyze context to improve PII detection confidence"""
    
    def __init__(self):
        # Context keywords that indicate PII likelihood
        self.context_keywords = {
            "personal": ["name", "address", "phone", "email", "personal", "identity"],
            "financial": ["card", "account", "bank", "payment", "credit", "ssn"],
            "corporate": ["employee", "staff", "worker", "id", "badge", "access"],
            "technical": ["ip", "address", "server", "system", "network", "mac"],
            "security": ["password", "credential", "key", "token", "auth", "login"]
        }
    
    def analyze_context(self, text: str, entity: PIIEntity) -> float:
        """Analyze context around entity to adjust confidence"""
        # Get context window around entity
        window_size = 100
        start = max(0, entity.start - window_size)
        end = min(len(text), entity.end + window_size)
        context = text[start:end].lower()
        
        context_score = 0.0
        max_boost = 0.2
        
        # Check for relevant context keywords
        for category, keywords in self.context_keywords.items():
            for keyword in keywords:
                if keyword in context:
                    if category == "personal" and entity.label in ["PERSON", "EMAIL_ADDRESS", "PHONE_NUMBER"]:
                        context_score += 0.05
                    elif category == "financial" and entity.label in ["CREDIT_CARD", "SSN"]:
                        context_score += 0.05
                    elif category == "corporate" and "EMPLOYEE" in entity.label:
                        context_score += 0.05
                    elif category == "technical" and entity.label in ["IP_ADDRESS", "MAC_ADDRESS"]:
                        context_score += 0.05
        
        return min(context_score, max_boost)

class MultiEnginePIIDetector:
    """Multi-engine PII detection system combining multiple approaches"""
    
    def __init__(self, 
                 engines: List[str] = None,
                 enable_context_analysis: bool = True,
                 confidence_threshold: float = 0.7):
        
        if engines is None:
            engines = ["presidio", "spacy", "transformers", "custom_patterns"]
        
        self.engines = {}
        self.enable_context_analysis = enable_context_analysis
        self.confidence_threshold = confidence_threshold
        
        # Initialize requested engines
        if "presidio" in engines:
            self.engines["presidio"] = PresidioPIIDetector()
        
        if "spacy" in engines:
            self.engines["spacy"] = SpacyPIIDetector()
        
        if "transformers" in engines:
            self.engines["transformers"] = TransformerPIIDetector()
        
        if "custom_patterns" in engines:
            self.engines["custom_patterns"] = SecurityPatternRecognizer()
        
        # Context analyzer
        if enable_context_analysis:
            self.context_analyzer = ContextAnalyzer()
    
    def detect_pii(self, text: str, language: str = "en") -> PIIAnalysisResult:
        """Detect PII using all configured engines"""
        start_time = time.time()
        
        all_entities = []
        confidence_scores = {}
        engines_used = []
        
        # Run detection with each engine
        for engine_name, engine in self.engines.items():
            try:
                if engine_name == "custom_patterns":
                    entities = engine.detect_patterns(text)
                elif engine_name == "presidio":
                    entities = engine.detect_entities(text, language)
                else:
                    entities = engine.detect_entities(text)
                
                all_entities.extend(entities)
                engines_used.append(engine_name)
                
                # Calculate average confidence for this engine
                if entities:
                    avg_conf = np.mean([e.confidence for e in entities])
                    confidence_scores[engine_name] = avg_conf
                
            except Exception as e:
                logger.error(f"Error in {engine_name} engine: {e}")
        
        # Enhance with context analysis
        if self.enable_context_analysis and hasattr(self, 'context_analyzer'):
            for entity in all_entities:
                context_score = self.context_analyzer.analyze_context(text, entity)
                entity.context_score = context_score
                entity.confidence = min(entity.confidence + context_score, 1.0)
        
        # Remove duplicates and merge overlapping entities
        merged_entities = self._merge_overlapping_entities(all_entities)
        
        # Filter by confidence threshold
        filtered_entities = [
            e for e in merged_entities 
            if e.confidence >= self.confidence_threshold
        ]
        
        # Sort by start position
        filtered_entities.sort(key=lambda x: x.start)
        
        processing_time = time.time() - start_time
        
        return PIIAnalysisResult(
            entities=filtered_entities,
            text=text,
            confidence_scores=confidence_scores,
            processing_time=processing_time,
            engines_used=engines_used,
            context_analysis={
                "total_entities_found": len(all_entities),
                "entities_after_merge": len(merged_entities),
                "entities_after_filter": len(filtered_entities),
                "average_confidence": np.mean([e.confidence for e in filtered_entities]) if filtered_entities else 0.0
            }
        )
    
    def _merge_overlapping_entities(self, entities: List[PIIEntity]) -> List[PIIEntity]:
        """Merge overlapping entities, keeping the one with higher confidence"""
        if not entities:
            return []
        
        # Sort by start position
        sorted_entities = sorted(entities, key=lambda x: x.start)
        merged = []
        
        for current in sorted_entities:
            if not merged:
                merged.append(current)
                continue
            
            last = merged[-1]
            
            # Check for overlap
            if current.start < last.end:
                # Overlapping entities - keep the one with higher confidence
                if current.confidence > last.confidence:
                    merged[-1] = current
                # If same confidence, keep the longer one
                elif (current.confidence == last.confidence and 
                      (current.end - current.start) > (last.end - last.start)):
                    merged[-1] = current
            else:
                merged.append(current)
        
        return merged
    
    def batch_detect(self, 
                    texts: List[str], 
                    max_workers: int = 4) -> List[PIIAnalysisResult]:
        """Process multiple texts in parallel"""
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(self.detect_pii, text) for text in texts]
            results = [future.result() for future in futures]
        
        return results
    
    def get_entity_statistics(self, results: List[PIIAnalysisResult]) -> Dict[str, Any]:
        """Generate statistics from multiple PII analysis results"""
        if not results:
            return {}
        
        all_entities = []
        for result in results:
            all_entities.extend(result.entities)
        
        if not all_entities:
            return {"total_entities": 0}
        
        # Count by label
        label_counts = {}
        for entity in all_entities:
            label_counts[entity.label] = label_counts.get(entity.label, 0) + 1
        
        # Count by engine
        engine_counts = {}
        for entity in all_entities:
            engine_counts[entity.engine] = engine_counts.get(entity.engine, 0) + 1
        
        # Calculate confidence statistics
        confidences = [e.confidence for e in all_entities]
        
        return {
            "total_entities": len(all_entities),
            "unique_labels": len(label_counts),
            "label_distribution": label_counts,
            "engine_distribution": engine_counts,
            "confidence_stats": {
                "mean": np.mean(confidences),
                "median": np.median(confidences),
                "std": np.std(confidences),
                "min": np.min(confidences),
                "max": np.max(confidences)
            },
            "sensitive_entities": len([e for e in all_entities if e.is_sensitive]),
            "average_processing_time": np.mean([r.processing_time for r in results])
        }

# Example usage and testing
if __name__ == "__main__":
    # Initialize multi-engine detector
    detector = MultiEnginePIIDetector(
        engines=["presidio", "spacy", "custom_patterns"],
        enable_context_analysis=True,
        confidence_threshold=0.7
    )
    
    # Test text with various PII types
    test_text = """
    Employee John Smith (ID: EMP123456) accessed the server room using badge B-445566.
    His email is john.smith@company.com and phone number is (555) 123-4567.
    The access was logged in system SYS-PROD-01 at IP address 192.168.1.100.
    Credit card 4532-1234-5678-9012 was used for the transaction.
    Policy POL-SEC-2024 requires two-factor authentication.
    """
    
    # Detect PII
    result = detector.detect_pii(test_text)
    
    # Print results
    print(f"Found {len(result.entities)} PII entities:")
    print("-" * 50)
    
    for entity in result.entities:
        print(f"Text: '{entity.text}'")
        print(f"Label: {entity.label}")
        print(f"Confidence: {entity.confidence:.2f}")
        print(f"Engine: {entity.engine}")
        print(f"Position: {entity.start}-{entity.end}")
        print(f"Sensitive: {entity.is_sensitive}")
        print("-" * 30)
    
    print(f"\nProcessing time: {result.processing_time:.2f}s")
    print(f"Engines used: {', '.join(result.engines_used)}")
    print(f"Context analysis: {result.context_analysis}")