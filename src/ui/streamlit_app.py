# Comprehensive Streamlit Application for SecureDoc AI
import streamlit as st
import cv2
import numpy as np
from PIL import Image
import pandas as pd
import json
import io
import base64
from typing import List, Dict, Any, Optional
import time
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import zipfile
from pathlib import Path

# Import core modules (adjust imports based on your project structure)
import sys
sys.path.append('src')

try:
    from core.document_processor import DocumentProcessor, DocumentProcessingResult
    from core.ocr_engine import HybridOCREngine
    from core.pii_detector import MultiEnginePIIDetector
    from core.logo_detector import LogoDetectionPipeline
    from core.anonymizer import AdvancedAnonymizer
    from utils.config import get_config, get_streamlit_config
except ImportError:
    st.error("Core modules not found. Please ensure all dependencies are installed.")
    st.stop()

# Page configuration
st.set_page_config(
    page_title="SecureDoc AI - Advanced Document Processing",
    page_icon="🔒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
.main-header {
    font-size: 3rem;
    font-weight: bold;
    text-align: center;
    background: linear-gradient(90deg, #FF6B6B, #4ECDC4);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 2rem;
}

.metric-card {
    background: white;
    padding: 1rem;
    border-radius: 10px;
    border-left: 4px solid #FF6B6B;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    margin: 0.5rem 0;
}

.success-box {
    background: #d4edda;
    border: 1px solid #c3e6cb;
    color: #155724;
    padding: 1rem;
    border-radius: 5px;
    margin: 1rem 0;
}

.warning-box {
    background: #fff3cd;
    border: 1px solid #ffeaa7;
    color: #856404;
    padding: 1rem;
    border-radius: 5px;
    margin: 1rem 0;
}

.error-box {
    background: #f8d7da;
    border: 1px solid #f5c6cb;
    color: #721c24;
    padding: 1rem;
    border-radius: 5px;
    margin: 1rem 0;
}
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'processing_history' not in st.session_state:
    st.session_state.processing_history = []
if 'current_result' not in st.session_state:
    st.session_state.current_result = None
if 'processor' not in st.session_state:
    st.session_state.processor = None

# Utility functions
@st.cache_data
def load_processor_config():
    """Load processor configuration"""
    return {
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

@st.cache_resource
def initialize_processor():
    """Initialize document processor with caching"""
    config = load_processor_config()
    return DocumentProcessor(config)

def create_download_link(data: bytes, filename: str, text: str):
    """Create download link for files"""
    b64 = base64.b64encode(data).decode()
    return f'<a href="data:application/octet-stream;base64,{b64}" download="{filename}" class="download-button">{text}</a>'

def display_processing_metrics(metrics, result):
    """Display processing metrics in a nice format"""
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="⏱️ Total Time",
            value=f"{metrics.total_time:.2f}s"
        )
    
    with col2:
        st.metric(
            label="🔍 Entities Found",
            value=metrics.entities_found
        )
    
    with col3:
        st.metric(
            label="🏷️ Logos Detected",
            value=metrics.logos_found
        )
    
    with col4:
        st.metric(
            label="✅ Success",
            value="Yes" if result.success else "No"
        )
    
    # Detailed timing breakdown
    st.subheader("⏱️ Processing Breakdown")
    
    timing_data = {
        'Stage': ['OCR', 'PII Detection', 'Logo Detection', 'Anonymization'],
        'Time (seconds)': [
            metrics.ocr_time,
            metrics.pii_detection_time,
            metrics.logo_detection_time,
            metrics.anonymization_time
        ]
    }
    
    fig = px.bar(
        timing_data,
        x='Stage',
        y='Time (seconds)',
        title="Processing Time by Stage",
        color='Time (seconds)',
        color_continuous_scale='viridis'
    )
    st.plotly_chart(fig, use_container_width=True)

def display_pii_analysis(entities):
    """Display PII analysis results"""
    if not entities:
        st.success("✅ No PII detected in the document!")
        return
    
    st.warning(f"⚠️ Found {len(entities)} PII entities")
    
    # Create DataFrame for display
    entity_data = []
    for entity in entities:
        entity_data.append({
            'Text': entity.text,
            'Type': entity.label,
            'Confidence': f"{entity.confidence:.0%}",
            'Engine': entity.engine,
            'Position': f"{entity.start}-{entity.end}",
            'Sensitive': "🔴" if entity.is_sensitive else "🟢"
        })
    
    df = pd.DataFrame(entity_data)
    st.dataframe(df, use_container_width=True)
    
    # Entity type distribution
    entity_types = {}
    for entity in entities:
        entity_types[entity.label] = entity_types.get(entity.label, 0) + 1
    
    if entity_types:
        fig = px.pie(
            values=list(entity_types.values()),
            names=list(entity_types.keys()),
            title="PII Entity Type Distribution"
        )
        st.plotly_chart(fig, use_container_width=True)

def display_logo_analysis(detections):
    """Display logo detection results"""
    if not detections:
        st.success("✅ No logos detected in the document!")
        return
    
    st.warning(f"🏷️ Found {len(detections)} logos")
    
    # Create DataFrame for display
    logo_data = []
    for detection in detections:
        logo_data.append({
            'Logo Type': detection.class_name,
            'Confidence': f"{detection.confidence:.0%}",
            'Area': f"{detection.area:.0f} px²",
            'Position': f"({detection.bbox[0]}, {detection.bbox[1]}) to ({detection.bbox[2]}, {detection.bbox[3]})"
        })
    
    df = pd.DataFrame(logo_data)
    st.dataframe(df, use_container_width=True)

def main():
    """Main Streamlit application"""
    
    # Header
    st.markdown('<h1 class="main-header">🔒 SecureDoc AI</h1>', unsafe_allow_html=True)
    st.markdown("**Advanced Document PII Detection & Logo Removal System**")
    
    # Initialize processor
    if st.session_state.processor is None:
        with st.spinner("Initializing AI engines..."):
            try:
                st.session_state.processor = initialize_processor()
                st.success("✅ All AI engines loaded successfully!")
            except Exception as e:
                st.error(f"❌ Failed to initialize processor: {e}")
                st.stop()
    
    # Sidebar configuration
    st.sidebar.header("⚙️ Configuration")
    
    # Processing options
    st.sidebar.subheader("OCR Settings")
    ocr_confidence = st.sidebar.slider(
        "OCR Confidence Threshold",
        0.0, 1.0, 0.7, 0.05,
        help="Minimum confidence for text extraction"
    )
    
    st.sidebar.subheader("PII Detection Settings")
    pii_confidence = st.sidebar.slider(
        "PII Confidence Threshold",
        0.0, 1.0, 0.8, 0.05,
        help="Minimum confidence for PII detection"
    )
    
    pii_engines = st.sidebar.multiselect(
        "PII Detection Engines",
        ["presidio", "spacy", "transformers", "custom_patterns"],
        default=["presidio", "spacy", "custom_patterns"]
    )
    
    st.sidebar.subheader("Logo Detection Settings")
    logo_confidence = st.sidebar.slider(
        "Logo Detection Confidence",
        0.0, 1.0, 0.6, 0.05,
        help="Minimum confidence for logo detection"
    )
    
    logo_removal_method = st.sidebar.selectbox(
        "Logo Removal Method",
        ["inpaint", "blur", "pixelate", "black_box"],
        help="Method for removing detected logos"
    )
    
    st.sidebar.subheader("Anonymization Settings")
    preserve_format = st.sidebar.checkbox("Preserve Format", True)
    realistic_replacements = st.sidebar.checkbox("Use Realistic Replacements", True)
    
    # Main content area
    tab1, tab2, tab3, tab4 = st.tabs([
        "📤 Document Processing", 
        "📊 Results Analysis", 
        "📈 Batch Processing",
        "📚 Processing History"
    ])
    
    with tab1:
        st.header("📤 Document Upload & Processing")
        
        # File upload
        uploaded_files = st.file_uploader(
            "Upload document images",
            type=['png', 'jpg', 'jpeg', 'tiff', 'bmp'],
            accept_multiple_files=True,
            help="Supported formats: PNG, JPG, JPEG, TIFF, BMP"
        )
        
        if uploaded_files:
            # Process single or multiple files
            if len(uploaded_files) == 1:
                st.subheader("Single Document Processing")
                
                uploaded_file = uploaded_files[0]
                
                # Display uploaded image
                image = Image.open(uploaded_file)
                st.image(image, caption=f"Uploaded: {uploaded_file.name}", use_column_width=True)
                
                # Processing button
                if st.button("🔄 Process Document", type="primary"):
                    # Create processing configuration
                    processing_config = {
                        "ocr": {"confidence_threshold": ocr_confidence},
                        "pii_detection": {
                            "engines": pii_engines,
                            "confidence_threshold": pii_confidence
                        },
                        "logo_detection": {
                            "confidence_threshold": logo_confidence,
                            "removal_method": logo_removal_method
                        },
                        "anonymization": {
                            "preserve_format": preserve_format,
                            "use_realistic_replacements": realistic_replacements
                        }
                    }
                    
                    # Convert PIL to numpy array
                    image_array = np.array(image)
                    if len(image_array.shape) == 3 and image_array.shape[2] == 3:
                        image_array = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
                    
                    # Process document
                    with st.spinner("Processing document... This may take a few moments."):
                        try:
                            result = st.session_state.processor.process_document(
                                image_array,
                                document_id=uploaded_file.name,
                                processing_options=processing_config
                            )
                            
                            st.session_state.current_result = result
                            
                            # Add to history
                            st.session_state.processing_history.append({
                                "filename": uploaded_file.name,
                                "timestamp": datetime.now(),
                                "result": result,
                                "success": result.success
                            })
                            
                        except Exception as e:
                            st.error(f"❌ Processing failed: {e}")
            
            else:
                st.subheader("Batch Document Processing")
                st.info(f"📁 {len(uploaded_files)} files selected for batch processing")
                
                # Batch processing button
                if st.button("🔄 Process All Documents", type="primary"):
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    batch_results = []
                    
                    for i, uploaded_file in enumerate(uploaded_files):
                        status_text.text(f"Processing {uploaded_file.name} ({i+1}/{len(uploaded_files)})")
                        
                        # Convert image
                        image = Image.open(uploaded_file)
                        image_array = np.array(image)
                        if len(image_array.shape) == 3 and image_array.shape[2] == 3:
                            image_array = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
                        
                        # Process document
                        try:
                            result = st.session_state.processor.process_document(
                                image_array,
                                document_id=uploaded_file.name
                            )
                            batch_results.append(result)
                            
                            # Add to history
                            st.session_state.processing_history.append({
                                "filename": uploaded_file.name,
                                "timestamp": datetime.now(),
                                "result": result,
                                "success": result.success
                            })
                            
                        except Exception as e:
                            st.error(f"❌ Failed to process {uploaded_file.name}: {e}")
                        
                        progress_bar.progress((i + 1) / len(uploaded_files))
                    
                    status_text.text("✅ Batch processing completed!")
                    
                    # Display batch results summary
                    successful = len([r for r in batch_results if r.success])
                    st.success(f"✅ Successfully processed {successful}/{len(batch_results)} documents")
                    
                    # Generate batch statistics
                    if batch_results:
                        stats = st.session_state.processor.get_processing_statistics(batch_results)
                        st.json(stats)
    
    with tab2:
        st.header("📊 Processing Results")
        
        if st.session_state.current_result is None:
            st.info("👆 Upload and process a document to see results here.")
        else:
            result = st.session_state.current_result
            
            if result.success:
                # Display metrics
                st.subheader("📈 Processing Metrics")
                display_processing_metrics(result.processing_metrics, result)
                
                # Results tabs
                result_tab1, result_tab2, result_tab3, result_tab4 = st.tabs([
                    "📝 Text Results",
                    "🔍 PII Analysis", 
                    "🏷️ Logo Analysis",
                    "🖼️ Images"
                ])
                
                with result_tab1:
                    st.subheader("Extracted Text")
                    if result.extracted_text:
                        st.text_area("Original Text", result.extracted_text, height=200)
                    else:
                        st.warning("No text extracted from the document.")
                    
                    st.subheader("Anonymized Text")
                    if result.final_text:
                        st.text_area("Anonymized Text", result.final_text, height=200)
                        
                        # Download options
                        st.subheader("📥 Download Results")
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            if st.download_button(
                                "📄 Download Original Text",
                                result.extracted_text,
                                file_name=f"{result.document_id}_original.txt",
                                mime="text/plain"
                            ):
                                st.success("Downloaded!")
                        
                        with col2:
                            if st.download_button(
                                "🔒 Download Anonymized Text",
                                result.final_text,
                                file_name=f"{result.document_id}_anonymized.txt",
                                mime="text/plain"
                            ):
                                st.success("Downloaded!")
                        
                        with col3:
                            # Create JSON report
                            report = {
                                "document_id": result.document_id,
                                "processing_time": result.processing_metrics.total_time,
                                "entities_found": len(result.detected_entities),
                                "logos_found": len(result.logo_detections),
                                "anonymization_mapping": result.anonymization_result.mapping if result.anonymization_result else {}
                            }
                            
                            if st.download_button(
                                "📊 Download Report",
                                json.dumps(report, indent=2),
                                file_name=f"{result.document_id}_report.json",
                                mime="application/json"
                            ):
                                st.success("Downloaded!")
                
                with result_tab2:
                    st.subheader("🔍 PII Detection Results")
                    display_pii_analysis(result.detected_entities)
                
                with result_tab3:
                    st.subheader("🏷️ Logo Detection Results")
                    display_logo_analysis(result.logo_detections)
                
                with result_tab4:
                    st.subheader("🖼️ Processed Images")
                    
                    img_col1, img_col2 = st.columns(2)
                    
                    with img_col1:
                        if result.original_image is not None:
                            st.image(
                                cv2.cvtColor(result.original_image, cv2.COLOR_BGR2RGB),
                                caption="Original Image",
                                use_column_width=True
                            )
                    
                    with img_col2:
                        if result.final_image is not None:
                            st.image(
                                cv2.cvtColor(result.final_image, cv2.COLOR_BGR2RGB),
                                caption="Processed Image",
                                use_column_width=True
                            )
            
            else:
                st.error(f"❌ Processing failed: {result.error_message}")
    
    with tab3:
        st.header("📈 Batch Processing Dashboard")
        
        if not st.session_state.processing_history:
            st.info("No processing history available. Process some documents first!")
        else:
            # Filter history for batch results
            history = st.session_state.processing_history
            
            # Summary statistics
            total_processed = len(history)
            successful = len([h for h in history if h["success"]])
            success_rate = (successful / total_processed) * 100 if total_processed > 0 else 0
            
            st.subheader("📊 Batch Summary")
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Processed", total_processed)
            with col2:
                st.metric("Successful", successful)
            with col3:
                st.metric("Success Rate", f"{success_rate:.1f}%")
            with col4:
                if history:
                    avg_time = np.mean([h["result"].processing_metrics.total_time for h in history if h["success"]])
                    st.metric("Avg Time", f"{avg_time:.2f}s")
            
            # Processing timeline
            if len(history) > 1:
                timeline_data = []
                for h in history:
                    timeline_data.append({
                        "Filename": h["filename"],
                        "Timestamp": h["timestamp"],
                        "Success": h["success"],
                        "Processing Time": h["result"].processing_metrics.total_time if h["success"] else 0
                    })
                
                df = pd.DataFrame(timeline_data)
                
                fig = px.scatter(
                    df,
                    x="Timestamp",
                    y="Processing Time",
                    color="Success",
                    hover_data=["Filename"],
                    title="Processing Timeline"
                )
                st.plotly_chart(fig, use_container_width=True)
    
    with tab4:
        st.header("📚 Processing History")
        
        if not st.session_state.processing_history:
            st.info("No processing history available.")
        else:
            # Display history table
            history_data = []
            for h in st.session_state.processing_history:
                history_data.append({
                    "Filename": h["filename"],
                    "Timestamp": h["timestamp"].strftime("%Y-%m-%d %H:%M:%S"),
                    "Status": "✅ Success" if h["success"] else "❌ Failed",
                    "Entities": len(h["result"].detected_entities) if h["success"] else 0,
                    "Logos": len(h["result"].logo_detections) if h["success"] else 0,
                    "Time (s)": f"{h['result'].processing_metrics.total_time:.2f}" if h["success"] else "N/A"
                })
            
            df = pd.DataFrame(history_data)
            st.dataframe(df, use_container_width=True)
            
            # Clear history button
            if st.button("🗑️ Clear History"):
                st.session_state.processing_history = []
                st.session_state.current_result = None
                st.experimental_rerun()
    
    # Footer
    st.markdown("---")
    st.markdown("""
    **🛠️ Built with:** Streamlit + EasyOCR + YOLOv8 + Microsoft Presidio + Custom NER  
    **🎯 Purpose:** Enterprise-grade document PII detection and logo removal  
    **⚡ Features:** OCR, multi-engine PII detection, YOLO logo detection, advanced anonymization
    """)

if __name__ == "__main__":
    main()