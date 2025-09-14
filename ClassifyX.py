import cv2
import os
import io
import time
import threading
import logging
import json
from typing import List, Tuple, Optional, Dict, Any
from datetime import datetime
import streamlit as st

import numpy as np

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from PIL import Image, ImageDraw, ImageEnhance, ImageFilter
from transformers import pipeline
import torch

import pyttsx3

# ---------------------------
# Enhanced Config / constants
# ---------------------------
MODEL_NAME = "google/vit-base-patch16-224"
UNCERTAIN_SAVE_DIR = "uncertain"
HISTORY_SAVE_DIR = "history" 
LOG_FILE = "demo.log"
HISTORY_FILE = "prediction_history.json"
MAX_LABELS = 10  # Increased to show more predictions
DISPLAY_SIZE = (512, 512)
CONFIDENCE_THRESHOLD = 0.6
MAX_HISTORY_ITEMS = 100

# Create directories
for dir_name in [UNCERTAIN_SAVE_DIR, HISTORY_SAVE_DIR]:
    os.makedirs(dir_name, exist_ok=True)

# Enhanced logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("cv_demo")

# ---------------------------
# Enhanced HTTP session with retries
# ---------------------------
def make_session(retries=5, backoff=0.3, status_forcelist=(500, 502, 503, 504)):
    """Create a robust HTTP session with retry logic."""
    s = requests.Session()
    retry = Retry(
        total=retries,
        backoff_factor=backoff,
        status_forcelist=status_forcelist,
        allowed_methods=["GET", "POST"]
    )
    adapter = HTTPAdapter(max_retries=retry)
    s.mount("http://", adapter)
    s.mount("https://", adapter)
    return s

SESSION = make_session()

# ---------------------------
# Enhanced TTS with voice selection
# ---------------------------
class TTSManager:
    def __init__(self):
        self.engine = None
        self.available_voices = []
        self.current_voice_idx = 0
        self.is_speaking = False
        
    def init_engine(self):
        """Initialize TTS engine with error handling."""
        if self.engine is None:
            try:
                self.engine = pyttsx3.init()
                self.engine.setProperty('rate', 150)
                self.engine.setProperty('volume', 0.9)
                
                # Get available voices
                voices = self.engine.getProperty('voices')
                self.available_voices = [v.id for v in voices] if voices else []
                
                if self.available_voices:
                    self.engine.setProperty('voice', self.available_voices[0])
                    
            except Exception as e:
                logger.error(f"TTS initialization failed: {e}")
                self.engine = None
    
    def speak_async(self, text: str):
        """Speak text asynchronously without blocking UI."""
        if self.is_speaking:
            return  # Don't overlap speech
            
        def _speak():
            try:
                self.is_speaking = True
                self.init_engine()
                if self.engine:
                    self.engine.say(text)
                    self.engine.runAndWait()
            except Exception as e:
                logger.error(f"TTS error: {e}")
            finally:
                self.is_speaking = False
        
        thread = threading.Thread(target=_speak, daemon=True)
        thread.start()
    
    def set_voice(self, voice_idx: int):
        """Change TTS voice."""
        if 0 <= voice_idx < len(self.available_voices):
            self.current_voice_idx = voice_idx
            if self.engine:
                self.engine.setProperty('voice', self.available_voices[voice_idx])

tts_manager = TTSManager()

# ---------------------------
# Enhanced Model Management
# ---------------------------
@st.cache_resource(show_spinner=False)
def load_model(model_name: str = MODEL_NAME):
    """Load model with device optimization."""
    try:
        device = 0 if torch.cuda.is_available() else -1
        device_name = "GPU" if device >= 0 else "CPU"
        
        with st.spinner(f"🔄 Loading model on {device_name} (first run downloads ~350MB)..."):
            classifier = pipeline(
                "image-classification", 
                model=model_name, 
                device=device,
                framework="pt"
            )
        
        st.success(f"✅ Model loaded successfully on {device_name}")
        return classifier
    except Exception as e:
        st.error(f"❌ Model loading failed: {e}")
        return None

def classify_image_enhanced(pil_img: Image.Image, top_k: int = MAX_LABELS) -> Optional[List[Dict]]:
    """Enhanced image classification with error handling."""
    try:
        model = load_model()
        if model is None:
            return None
            
        # Preprocess image for better results
        if pil_img.size[0] < 224 or pil_img.size[1] < 224:
            pil_img = pil_img.resize((224, 224), Image.LANCZOS)
        
        results = model(pil_img, top_k=top_k)
        
        # Add confidence categories
        for result in results:
            score = result['score']
            if score >= 0.8:
                result['confidence_level'] = "High"
            elif score >= 0.6:
                result['confidence_level'] = "Medium" 
            else:
                result['confidence_level'] = "Low"
        
        return results
    except Exception as e:
        logger.error(f"Classification error: {e}")
        st.error(f"❌ Classification failed: {e}")
        return None

# ---------------------------
# Enhanced Image Processing
# ---------------------------
class ImageProcessor:
    @staticmethod
    def resize_for_display(img: Image.Image, size=DISPLAY_SIZE) -> Image.Image:
        """Smart resize maintaining aspect ratio."""
        img_ratio = img.width / img.height
        target_ratio = size[0] / size[1]
        
        if img_ratio > target_ratio:
            new_width = size[0]
            new_height = int(size[0] / img_ratio)
        else:
            new_width = int(size[1] * img_ratio)
            new_height = size[1]
            
        return img.resize((new_width, new_height), Image.LANCZOS)
    
    @staticmethod
    def enhance_image(img: Image.Image, brightness=1.0, contrast=1.0, sharpness=1.0) -> Image.Image:
        """Apply image enhancements."""
        if brightness != 1.0:
            img = ImageEnhance.Brightness(img).enhance(brightness)
        if contrast != 1.0:
            img = ImageEnhance.Contrast(img).enhance(contrast)
        if sharpness != 1.0:
            img = ImageEnhance.Sharpness(img).enhance(sharpness)
        return img
    
    @staticmethod
    def apply_filter(img: Image.Image, filter_type: str) -> Image.Image:
        """Apply various filters to image."""
        filters = {
            "blur": ImageFilter.BLUR,
            "contour": ImageFilter.CONTOUR,
            "detail": ImageFilter.DETAIL,
            "edge_enhance": ImageFilter.EDGE_ENHANCE,
            "emboss": ImageFilter.EMBOSS,
            "sharpen": ImageFilter.SHARPEN
        }
        
        if filter_type in filters:
            return img.filter(filters[filter_type])
        return img

image_processor = ImageProcessor()

# ---------------------------
# Enhanced URL fetching
# ---------------------------
def fetch_image_from_url(url: str, timeout=15) -> Optional[Image.Image]:
    """Enhanced image fetching with better error handling."""
    try:
        # Validate URL format
        if not (url.startswith('http://') or url.startswith('https://')):
            return None
            
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Accept': 'image/webp,image/apng,image/jpeg,image/png,image/*,*/*;q=0.8',
            'Accept-Encoding': 'gzip, deflate',
            'Accept-Language': 'en-US,en;q=0.5',
            'Connection': 'keep-alive',
        }
        
        with st.spinner("🌐 Fetching image from URL..."):
            resp = SESSION.get(url, headers=headers, timeout=timeout, stream=True)
            resp.raise_for_status()
            
            # Check content type
            content_type = resp.headers.get('content-type', '').lower()
            if 'image' not in content_type:
                st.warning(f"⚠️ URL doesn't seem to be an image (content-type: {content_type})")
                return None
            
            # Check file size (limit to 50MB)
            content_length = resp.headers.get('content-length')
            if content_length and int(content_length) > 50 * 1024 * 1024:
                st.error("❌ Image too large (>50MB)")
                return None
            
            bio = io.BytesIO(resp.content)
            img = Image.open(bio).convert('RGB')
            
            # Validate image dimensions
            if img.size[0] < 32 or img.size[1] < 32:
                st.warning("⚠️ Image too small (minimum 32x32 pixels)")
                return None
                
            logger.info(f"Successfully fetched image: {img.size}")
            return img
            
    except requests.exceptions.RequestException as e:
        st.error(f"❌ Network error: {str(e)}")
        logger.error(f"Fetch error for {url}: {e}")
    except Exception as e:
        st.error(f"❌ Error processing image: {str(e)}")
        logger.error(f"Image processing error for {url}: {e}")
    
    return None

# ---------------------------
# Enhanced History Management
# ---------------------------
class HistoryManager:
    def __init__(self):
        self.load_history()
    
    def load_history(self):
        """Load prediction history from file."""
        try:
            if os.path.exists(HISTORY_FILE):
                with open(HISTORY_FILE, 'r') as f:
                    st.session_state['history'] = json.load(f)
            else:
                st.session_state['history'] = []
        except Exception as e:
            logger.error(f"Error loading history: {e}")
            st.session_state['history'] = []
    
    def save_history(self):
        """Save prediction history to file."""
        try:
            with open(HISTORY_FILE, 'w') as f:
                json.dump(st.session_state['history'][:MAX_HISTORY_ITEMS], f, indent=2)
        except Exception as e:
            logger.error(f"Error saving history: {e}")
    
    def add_prediction(self, source: str, results: List[Dict], img_path: str = None):
        """Add new prediction to history."""
        entry = {
            'timestamp': datetime.now().isoformat(),
            'time': time.strftime("%H:%M:%S"),
            'source': source,
            'top_prediction': results[0] if results else {},
            'all_predictions': results[:5],  # Store top 5
            'image_path': img_path
        }
        
        st.session_state['history'].insert(0, entry)
        # Keep only recent entries
        st.session_state['history'] = st.session_state['history'][:MAX_HISTORY_ITEMS]
        self.save_history()

history_manager = HistoryManager()

# ---------------------------
# Enhanced saving functions
# ---------------------------
def save_prediction_image(img: Image.Image, results: List[Dict], prefix="prediction"):
    """Save image with prediction metadata."""
    try:
        timestamp = int(time.time())
        top_label = results[0]['label'].replace(' ', '_').replace('/', '_')
        confidence = int(results[0]['score'] * 100)
        
        filename = f"{prefix}_{top_label}_{confidence}pct_{timestamp}.jpg"
        filepath = os.path.join(HISTORY_SAVE_DIR, filename)
        
        img.save(filepath, quality=85)
        logger.info(f"Saved prediction image: {filepath}")
        return filepath
    except Exception as e:
        logger.error(f"Error saving image: {e}")
        return None

# ---------------------------
# Enhanced result display
# ---------------------------
def create_results_display(results: List[Dict]) -> str:
    """Create enhanced visual results display."""
    if not results:
        return "No results available"
    
    lines = ["### 🎯 Classification Results\n"]
    
    max_bar_length = 25
    for i, result in enumerate(results):
        label = result.get('label', 'Unknown')
        score = result.get('score', 0.0)
        confidence_level = result.get('confidence_level', 'Unknown')
        
        # Create progress bar
        filled_length = int(score * max_bar_length)
        bar = "█" * filled_length + "░" * (max_bar_length - filled_length)
        
        # Color coding based on confidence
        if confidence_level == "High":
            color = "🟢"
        elif confidence_level == "Medium":
            color = "🟡"
        else:
            color = "🔴"
        
        # Rank indicator
        rank = "🥇" if i == 0 else "🥈" if i == 1 else "🥉" if i == 2 else f"#{i+1}"
        
        line = f"{rank} {color} `{bar}` **{label}** ({score:.1%}) - *{confidence_level} confidence*"
        lines.append(line)
    
    return "\n\n".join(lines)

# ---------------------------
# Enhanced Streamlit UI
# ---------------------------
def main():
    st.set_page_config(
        page_title="🖼️ Advanced CV Demo", 
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title("🖼️ Advanced Computer Vision Demo")
    st.markdown("*Powered by Vision Transformer (ViT) - Real-time image classification with enhanced features*")

    # Enhanced Sidebar
    with st.sidebar:
        st.header("🎛️ Demo Controls")
        
        # TTS Settings
        st.subheader("🔊 Text-to-Speech")
        speak_toggle = st.checkbox("Enable voice output", value=True)
        
        if speak_toggle and tts_manager.available_voices:
            voice_names = [f"Voice {i+1}" for i in range(len(tts_manager.available_voices))]
            selected_voice = st.selectbox("Select voice", voice_names)
            if selected_voice:
                voice_idx = voice_names.index(selected_voice)
                tts_manager.set_voice(voice_idx)
        
        st.markdown("---")
        
        # Model Settings
        st.subheader("🧠 Model Settings")
        confidence_threshold = st.slider("Confidence threshold", 0.1, 1.0, CONFIDENCE_THRESHOLD, 0.1)
        max_predictions = st.slider("Max predictions to show", 1, 15, MAX_LABELS)
        
        st.markdown("---")
        
        # Quick Test URLs
        st.subheader("🔗 Quick Test URLs")
        test_urls = [
            "https://placekitten.com/400/400",
            "https://place.dog/400/400", 
            "https://picsum.photos/400/400",
            "https://source.unsplash.com/400x400/?animal",
            "https://source.unsplash.com/400x400/?food"
        ]
        
        for url in test_urls:
            if st.button(f"📎 {url.split('/')[-2]}", key=url):
                st.session_state['quick_url'] = url

        st.markdown("---")
        
        # Statistics
        st.subheader("📊 Session Stats")
        if 'history' in st.session_state:
            total_predictions = len(st.session_state['history'])
            st.metric("Total predictions", total_predictions)
            
            if total_predictions > 0:
                high_conf = sum(1 for h in st.session_state['history'] 
                              if h.get('top_prediction', {}).get('score', 0) >= 0.8)
                st.metric("High confidence", f"{high_conf}/{total_predictions}")

    # Main content area
    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("📤 Input Source")
        
        # Input tabs for better organization
        tab1, tab2, tab3 = st.tabs(["📁 Upload", "📷 Camera", "🌐 URL"])
        
        with tab1:
            uploaded = st.file_uploader(
                "Choose an image file", 
                type=["jpg", "jpeg", "png", "bmp", "tiff"],
                help="Supported formats: JPG, PNG, BMP, TIFF"
            )
        
        with tab2:
            cam = st.camera_input("Take a photo")
        
        with tab3:
            # Check for quick URL selection
            default_url = st.session_state.get('quick_url', '')
            test_url = st.text_input(
                "Image URL", 
                value=default_url,
                placeholder="https://example.com/image.jpg"
            )
            if st.session_state.get('quick_url'):
                del st.session_state['quick_url']  # Clear after use
        
        # Image enhancement controls
        if uploaded or cam or test_url:
            st.subheader("🎨 Image Enhancement")
            with st.expander("Adjust image properties"):
                brightness = st.slider("Brightness", 0.5, 2.0, 1.0, 0.1)
                contrast = st.slider("Contrast", 0.5, 2.0, 1.0, 0.1)
                sharpness = st.slider("Sharpness", 0.5, 2.0, 1.0, 0.1)
                
                filter_type = st.selectbox(
                    "Apply filter",
                    ["none", "blur", "sharpen", "edge_enhance", "emboss"]
                )

    with col2:
        st.subheader("🎯 Analysis Results")
        
        # Determine image source
        pil_img = None
        source_desc = ""
        
        if uploaded is not None:
            pil_img = Image.open(uploaded).convert('RGB')
            source_desc = f"uploaded file: {uploaded.name}"
        elif cam is not None:
            pil_img = Image.open(cam).convert('RGB') 
            source_desc = "camera capture"
        elif test_url.strip():
            pil_img = fetch_image_from_url(test_url.strip())
            source_desc = f"URL: {test_url.strip()}"

        # Process and analyze image
        if pil_img:
            # Apply enhancements
            if 'brightness' in locals():
                pil_img = image_processor.enhance_image(
                    pil_img, brightness, contrast, sharpness
                )
            
            if 'filter_type' in locals() and filter_type != "none":
                pil_img = image_processor.apply_filter(pil_img, filter_type)
            
            # Display image
            display_img = image_processor.resize_for_display(pil_img)
            st.image(display_img, caption=f"📷 {source_desc}", use_column_width=True)
            
            # Classify image
            with st.spinner("🔍 Analyzing image..."):
                results = classify_image_enhanced(pil_img, top_k=max_predictions)
            
            if results:
                # Display results
                st.markdown(create_results_display(results))
                
                # Save image if confidence is low
                img_path = None
                if results[0]['score'] < confidence_threshold:
                    img_path = save_prediction_image(pil_img, results, "uncertain")
                    st.info(f"💾 Low confidence detection saved for review")
                else:
                    img_path = save_prediction_image(pil_img, results, "confident")
                
                # Text-to-speech
                if speak_toggle:
                    top_result = results[0]
                    confidence_text = top_result['confidence_level'].lower()
                    speech_text = (f"I see {top_result['label']} with "
                                 f"{confidence_text} confidence at "
                                 f"{top_result['score']*100:.0f} percent")
                    tts_manager.speak_async(speech_text)
                
                # Add to history
                history_manager.add_prediction(source_desc, results, img_path)
                
                # Export results
                with st.expander("📊 Export Results"):
                    results_json = json.dumps(results, indent=2)
                    st.download_button(
                        "Download JSON",
                        results_json,
                        f"prediction_{int(time.time())}.json",
                        "application/json"
                    )
            else:
                st.error("❌ Failed to analyze image. Please try again.")
        
        else:
            st.info("👆 Select an input method above to start analyzing images")

    # Enhanced History Section
    st.markdown("---")
    st.subheader("📜 Prediction History")
    
    if st.session_state.get('history'):
        # History controls
        col_hist1, col_hist2, col_hist3 = st.columns([1, 1, 1])
        
        with col_hist1:
            show_count = st.selectbox("Show recent", [10, 25, 50, "all"])
        
        with col_hist2:
            if st.button("🗑️ Clear History"):
                st.session_state['history'] = []
                history_manager.save_history()
                st.experimental_rerun()
        
        with col_hist3:
            if st.button("📥 Download History"):
                history_json = json.dumps(st.session_state['history'], indent=2)
                st.download_button(
                    "Download",
                    history_json,
                    f"history_{int(time.time())}.json",
                    "application/json"
                )
        
        # Display history
        history_to_show = st.session_state['history']
        if show_count != "all":
            history_to_show = history_to_show[:int(show_count)]
        
        for i, item in enumerate(history_to_show):
            with st.expander(f"🕒 {item['time']} - {item.get('top_prediction', {}).get('label', 'Unknown')} "
                           f"({item.get('top_prediction', {}).get('score', 0)*100:.1f}%)"):
                st.write(f"**Source:** {item['source']}")
                st.write(f"**Timestamp:** {item['timestamp']}")
                
                if 'all_predictions' in item and item['all_predictions']:
                    st.write("**Top predictions:**")
                    for j, pred in enumerate(item['all_predictions'][:3]):
                        st.write(f"{j+1}. {pred['label']} - {pred['score']*100:.1f}%")
    
    else:
        st.write("No predictions yet. Upload an image to get started! 🚀")

    # Footer
    st.markdown("---")
    st.markdown("*Made with ❤️ using Streamlit and HuggingFace Transformers*")

if __name__ == "__main__":
    main()
