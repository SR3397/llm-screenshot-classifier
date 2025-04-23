import os
import torch
import numpy as np
from tqdm import tqdm
import shutil
from pathlib import Path
import logging
import argparse
import json
from transformers import AutoTokenizer, AutoModelForCausalLM
from PIL import Image
from llama_cpp import Llama
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("screenshot_classifier.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("ScreenshotClassifier")

class ScreenshotClassifier:
    def __init__(self, input_dir, output_base_dir, batch_size=8, device=None, sensitivity="moderate", 
                 llm_model="TheBloke/Llama-3-8B-Instruct-GPTQ", ocr_engine="paddle",
                 resize_images=True, max_width=1920, max_height=1080,
                 ocr_gpu_fallback=True, ocr_gpu_mem=8192,
                 llm_threads=4, llm_batch_size=4):
        """
        Initialize the classifier
        
        Args:
            input_dir: Directory containing the screenshots
            output_base_dir: Base directory to save sorted screenshots
            batch_size: Batch size for processing
            device: Device to use (cuda or cpu)
            sensitivity: Sensitivity level for classification (low, moderate, high)
            llm_model: HuggingFace model ID or path to GGUF file for the LLM
            ocr_engine: OCR engine to use ('tesseract', 'easyocr', or 'paddle')
            resize_images: Whether to resize large images before OCR
            max_width: Maximum width for resizing
            max_height: Maximum height for resizing
            ocr_gpu_fallback: Whether to fall back to CPU if GPU fails for OCR
            ocr_gpu_mem: GPU memory allocation for PaddleOCR in MB
            llm_threads: Number of threads for parallel LLM processing
            llm_batch_size: Batch size for LLM classification
        """
        self.input_dir = Path(input_dir)
        self.output_base_dir = Path(output_base_dir)
        self.batch_size = batch_size
        self.sensitivity = sensitivity
        self.llm_model = llm_model
        self.ocr_engine = ocr_engine.lower()
        self.resize_images = resize_images
        self.max_width = max_width
        self.max_height = max_height
        self.ocr_gpu_fallback = ocr_gpu_fallback
        self.ocr_gpu_mem = ocr_gpu_mem
        self.llm_threads = llm_threads
        self.llm_batch_size = llm_batch_size
        
        # Set device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        logger.info(f"Using device: {self.device}")
        logger.info(f"OCR batch size: {self.batch_size}")
        logger.info(f"LLM threads: {self.llm_threads}")
        logger.info(f"LLM batch size: {self.llm_batch_size}")
        
        # Create output directories
        self.sfw_dir = self.output_base_dir / "SFW"
        self.unsafe_dir = self.output_base_dir / "Unsafe"
        self.nsfw_dir = self.output_base_dir / "NSFW"
        
        os.makedirs(self.sfw_dir, exist_ok=True)
        os.makedirs(self.unsafe_dir, exist_ok=True)
        os.makedirs(self.nsfw_dir, exist_ok=True)
        
        # Temp directory for processed images
        self.temp_dir = Path("./temp_images")
        self.temp_dir.mkdir(exist_ok=True)
        
        # Performance tracking
        self.ocr_time = 0
        self.llm_time = 0
        self.num_processed = 0
        
        # LLM thread lock (for thread safety)
        self.llm_lock = threading.Lock()
        
        # Initialize models
        self.init_models()
    
    def init_models(self):
        """Initialize all required models"""
        # First, initialize OCR engine
        self.init_ocr_engine()
        
        # Then, initialize LLM
        self.init_llm_model()
            
        logger.info("Models loaded successfully")
    
    def init_ocr_engine(self):
        """Initialize the selected OCR engine"""
        logger.info(f"Initializing OCR engine: {self.ocr_engine}")
        
        if self.ocr_engine == 'tesseract':
            try:
                import pytesseract
                self.ocr = pytesseract
                # Check if tesseract is installed
                try:
                    self.ocr.get_tesseract_version()
                    logger.info(f"Tesseract OCR initialized successfully, version: {self.ocr.get_tesseract_version()}")
                except Exception as e:
                    logger.warning(f"Tesseract executable not found. Error: {str(e)}")
                    logger.warning("You may need to set the path to the tesseract executable.")
                    # Try to find the tesseract executable
                    for tesseract_path in [
                        r"C:\Program Files\Tesseract-OCR\tesseract.exe",
                        r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe"
                    ]:
                        if Path(tesseract_path).exists():
                            self.ocr.pytesseract.tesseract_cmd = tesseract_path
                            logger.info(f"Found Tesseract at: {tesseract_path}")
                            break
            except ImportError:
                logger.error("Pytesseract not installed. Install with 'pip install pytesseract' and ensure Tesseract OCR is installed on your system.")
                raise
                
        elif self.ocr_engine == 'easyocr':
            try:
                import easyocr
                gpu_flag = self.device.type=='cuda'
                logger.info(f"Initializing EasyOCR with GPU: {gpu_flag}")
                try:
                    self.ocr = easyocr.Reader(['en'], gpu=gpu_flag)
                    logger.info("EasyOCR initialized successfully with GPU")
                except Exception as e:
                    if self.ocr_gpu_fallback and gpu_flag:
                        logger.warning(f"Failed to initialize EasyOCR with GPU: {str(e)}")
                        logger.warning("Falling back to CPU mode")
                        self.ocr = easyocr.Reader(['en'], gpu=False)
                        logger.info("EasyOCR initialized successfully with CPU")
                    else:
                        raise
            except ImportError:
                logger.error("EasyOCR not installed. Install with 'pip install easyocr'.")
                raise
                
        elif self.ocr_engine == 'paddle':
            try:
                from paddleocr import PaddleOCR
                # First try with GPU
                use_gpu = self.device.type=='cuda'
                logger.info(f"Initializing PaddleOCR with GPU: {use_gpu}, GPU Memory: {self.ocr_gpu_mem}MB")
                
                try:
                    # Initialize with GPU memory allocation and batch size
                    self.ocr = PaddleOCR(
                        use_angle_cls=True, 
                        lang='en', 
                        use_gpu=use_gpu,
                        gpu_mem=self.ocr_gpu_mem,
                        page_num=self.batch_size,  # This is PaddleOCR's batch size parameter
                        det_db_unclip_ratio=2.0,  # Optimize for screen text
                        rec_batch_num=32,  # Recognition batch size
                        enable_mkldnn=False if use_gpu else True  # Enable MKL-DNN on CPU for better performance
                    )
                    # Test the OCR on a small image to verify it works
                    test_img = np.ones((32, 100, 3), dtype=np.uint8) * 255
                    test_path = str(self.temp_dir / "test.png")
                    Image.fromarray(test_img).save(test_path)
                    self.ocr.ocr(test_path)
                    logger.info(f"PaddleOCR initialized successfully with GPU: {use_gpu}")
                except Exception as e:
                    if self.ocr_gpu_fallback and use_gpu:
                        logger.warning(f"Failed to initialize PaddleOCR with GPU: {str(e)}")
                        logger.warning("Falling back to CPU mode")
                        self.ocr = PaddleOCR(
                            use_angle_cls=True, 
                            lang='en', 
                            use_gpu=False,
                            page_num=self.batch_size,
                            enable_mkldnn=True  # Enable MKL-DNN on CPU for better performance
                        )
                        logger.info("PaddleOCR initialized successfully with CPU")
                    else:
                        raise
            except ImportError:
                logger.error("PaddleOCR not installed. Install with 'pip install paddleocr'.")
                raise
                
        else:
            logger.error(f"Unknown OCR engine: {self.ocr_engine}")
            raise ValueError(f"Unknown OCR engine: {self.ocr_engine}. Choose from: tesseract, easyocr, paddle")
    
    def init_llm_model(self):
        """Initialize the LLM for content classification"""
        logger.info("Loading LLM for content classification...")
        
        try:
            # Check if the model is a GGUF file or path to a GGUF file
            if isinstance(self.llm_model, str) and (self.llm_model.endswith('.gguf') or os.path.exists(self.llm_model) and Path(self.llm_model).suffix.lower() == '.gguf'):
                logger.info("Loading GGUF model with llama-cpp-python...")
                # Load the model with llama-cpp-python
                self.llm = Llama(
                    model_path=self.llm_model,
                    n_ctx=1024,  # Reduced context window size for better performance
                    n_gpu_layers=-1,  # Use all GPU layers if possible
                    n_threads=self.llm_threads  # Set number of CPU threads
                )
                self.model_type = "gguf"
            else:
                logger.info("Loading HuggingFace model with transformers...")
                # Load tokenizer and model for content classification using HuggingFace transformers
                self.tokenizer = AutoTokenizer.from_pretrained(self.llm_model)
                
                # Load the model with quantization settings for efficiency
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.llm_model,
                    torch_dtype=torch.float16,
                    device_map="auto",
                    load_in_8bit=True,  # 8-bit quantization
                )
                self.model_type = "hf"
        except Exception as e:
            logger.error(f"Error loading LLM: {str(e)}")
            raise
    
    def preprocess_image(self, image_path):
        """Preprocess image before OCR if needed"""
        if not self.resize_images:
            return image_path
            
        try:
            img = Image.open(image_path)
            # Only resize if larger than max dimensions
            if img.width > self.max_width or img.height > self.max_height:
                img.thumbnail((self.max_width, self.max_height), Image.LANCZOS)
                
                # Save to temp file with same format
                # Use original filename but in temp dir
                temp_path = self.temp_dir / Path(image_path).name
                img.save(temp_path)
                return str(temp_path)
            
            return image_path
        except Exception as e:
            logger.error(f"Error preprocessing image {image_path}: {str(e)}")
            return image_path
        
    def extract_text_batch(self, image_paths):
        """Extract text from a batch of images using the selected OCR engine"""
        start_time = time.time()
        results = []
        
        try:
            # Preprocess images if needed
            processed_paths = [self.preprocess_image(img_path) for img_path in image_paths]
            
            if self.ocr_engine == 'tesseract':
                # Tesseract doesn't support native batching, use ThreadPoolExecutor
                with ThreadPoolExecutor(max_workers=min(self.batch_size, 8)) as executor:
                    futures = []
                    for path in processed_paths:
                        futures.append(executor.submit(self.extract_text_single, path))
                    
                    for future, original_path in zip(as_completed(futures), image_paths):
                        results.append((original_path, future.result()))
                
            elif self.ocr_engine == 'easyocr':
                # Process each image individually to avoid potential batch issues
                with ThreadPoolExecutor(max_workers=min(self.batch_size, 8)) as executor:
                    futures = []
                    for path in processed_paths:
                        futures.append(executor.submit(self.extract_text_single, path))
                    
                    for future, original_path in zip(as_completed(futures), image_paths):
                        results.append((original_path, future.result()))
                
            elif self.ocr_engine == 'paddle':
                # Process images individually for better error handling
                with ThreadPoolExecutor(max_workers=min(self.batch_size, 4)) as executor:
                    futures = []
                    for path in processed_paths:
                        futures.append(executor.submit(self.extract_text_single, path))
                    
                    for future, original_path in zip(as_completed(futures), image_paths):
                        results.append((original_path, future.result()))
            
            self.ocr_time += time.time() - start_time
            return results
        
        except Exception as e:
            logger.error(f"Error in batch OCR processing: {str(e)}")
            # Return empty results for all images in the batch
            return [(img_path, "") for img_path in image_paths]
    
    def extract_text_single(self, image_path):
        """Extract text from a single image using the selected OCR engine"""
        try:
            if self.ocr_engine == 'tesseract':
                # Use pytesseract
                img = Image.open(image_path)
                extracted_text = self.ocr.image_to_string(img)
                
            elif self.ocr_engine == 'easyocr':
                # Use EasyOCR
                result = self.ocr.readtext(image_path)
                extracted_text = " ".join([item[1] for item in result])
                
            elif self.ocr_engine == 'paddle':
                # Use PaddleOCR with better error handling
                try:
                    result = self.ocr.ocr(image_path, cls=True)
                    # PaddleOCR returns a list of results for each image
                    extracted_text = ""
                    if result and len(result) > 0:
                        for line in result[0]:
                            if line and len(line) >= 2:
                                extracted_text += line[1][0] + " "
                except Exception as e:
                    logger.warning(f"Error in PaddleOCR for {image_path}: {str(e)}")
                    # Try again with different parameters
                    try:
                        result = self.ocr.ocr(image_path, cls=False)  # Disable angle classifier
                        extracted_text = ""
                        if result and len(result) > 0:
                            for line in result[0]:
                                if line and len(line) >= 2:
                                    extracted_text += line[1][0] + " "
                    except Exception:
                        # Last resort: open the image and see if it's valid
                        try:
                            img = Image.open(image_path)
                            img.thumbnail((100, 100))  # Just to verify it's a valid image
                            logger.warning(f"Image {image_path} is valid but OCR failed. Returning empty text.")
                            return ""
                        except Exception:
                            logger.error(f"Image {image_path} is invalid.")
                            return ""
            
            return extracted_text.strip()
        except Exception as e:
            logger.error(f"Error extracting text from {image_path}: {str(e)}")
            return ""
    
    def classify_content(self, extracted_text):
        """
        Classify the content using the LLM
        
        Returns:
            category: 'SFW', 'Unsafe', or 'NSFW'
            confidence: Confidence score for the classification
        """
        start_time = time.time()
        
        # Skip classification if text is empty or very short
        if not extracted_text or len(extracted_text) < 5:
            self.llm_time += time.time() - start_time
            return "SFW", 0.9
        
        # Simplify the prompt for faster processing
        prompt = f"""Classify this screenshot text as "SFW", "Unsafe", or "NSFW":
- SFW: Safe for workplace
- Unsafe: Sensitive but not explicit (violence, drugs, etc.)
- NSFW: Adult/sexual content

Sensitivity: {self.sensitivity}

Text: "{extracted_text}"

Format: {{"category": "SFW|Unsafe|NSFW"}}"""

        try:
            # Use lock to prevent concurrent access to the LLM
            with self.llm_lock:
                # Generate response based on model type
                if self.model_type == "gguf":
                    # Use llama-cpp-python for GGUF models
                    response = self.llm(
                        prompt,
                        max_tokens=50,  # Reduced token count
                        temperature=0.1,
                        stop=["\n\n"],
                        echo=False
                    )
                    response_text = response["choices"][0]["text"]
                else:
                    # Use transformers for HuggingFace models
                    inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
                    
                    with torch.no_grad():
                        output = self.model.generate(
                            **inputs,
                            max_new_tokens=50,
                            do_sample=False,
                            temperature=0.1,
                            pad_token_id=self.tokenizer.eos_token_id
                        )
                    
                    response_text = self.tokenizer.decode(output[0], skip_special_tokens=True)
            
            # Extract the category (optimized for speed)
            response_lower = response_text.lower()
            
            # Quick classification based on keyword matching
            if "nsfw" in response_lower:
                category, confidence = "NSFW", 0.8
            elif "unsafe" in response_lower:
                category, confidence = "Unsafe", 0.7
            else:
                # Try to extract JSON
                try:
                    json_start = response_text.find('{')
                    json_end = response_text.rfind('}') + 1
                    
                    if json_start != -1 and json_end != -1:
                        json_response = response_text[json_start:json_end]
                        result = json.loads(json_response)
                        
                        category = result.get('category', 'SFW')
                        confidence = result.get('confidence', 0.6)
                    else:
                        category, confidence = "SFW", 0.6
                except Exception:
                    category, confidence = "SFW", 0.6
            
        except Exception as e:
            logger.error(f"Error classifying content: {str(e)}")
            category, confidence = "Unsafe", 0.5  # Default to Unsafe when errors occur
        
        self.llm_time += time.time() - start_time
        return category, confidence
    
    def classify_batch(self, batch_data):
        """Classify a batch of texts and return results"""
        results = []
        for img_path, text in batch_data:
            category, confidence = self.classify_content(text)
            results.append((img_path, text, category, confidence))
        return results
    
    def process_screenshots(self):
        """Process all screenshots in the input directory using pipeline processing"""
        # Get list of image files
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.gif']
        image_files = []
        
        for ext in image_extensions:
            image_files.extend(list(self.input_dir.glob(f"*{ext}")))
        
        total_files = len(image_files)
        logger.info(f"Found {total_files} files to process")
        
        # Process in batches
        ocr_batch_size = self.batch_size
        llm_batch_size = self.llm_batch_size
        
        ocr_batches = [image_files[i:i + ocr_batch_size] for i in range(0, len(image_files), ocr_batch_size)]
        
        start_time = time.time()
        
        with tqdm(total=total_files, desc="Processing screenshots") as pbar:
            # Pipeline: OCR then Classification
            for ocr_batch in ocr_batches:
                try:
                    # Step 1: Extract text from batch of images
                    ocr_results = self.extract_text_batch(ocr_batch)
                    
                    # Step 2: Classify in sub-batches for better parallelism
                    llm_batches = [ocr_results[i:i + llm_batch_size] for i in range(0, len(ocr_results), llm_batch_size)]
                    
                    # Process each LLM batch with ThreadPoolExecutor if multiple threads
                    if self.llm_threads > 1 and len(llm_batches) > 1:
                        with ThreadPoolExecutor(max_workers=min(len(llm_batches), 4)) as executor:
                            futures = []
                            for llm_batch in llm_batches:
                                futures.append(executor.submit(self.classify_batch, llm_batch))
                            
                            for future in as_completed(futures):
                                batch_results = future.result()
                                self.process_classification_results(batch_results, pbar)
                    else:
                        # Process sequentially
                        for llm_batch in llm_batches:
                            batch_results = self.classify_batch(llm_batch)
                            self.process_classification_results(batch_results, pbar)
                
                except Exception as e:
                    logger.error(f"Error processing batch: {str(e)}")
                    pbar.update(len(ocr_batch))
        
        total_time = time.time() - start_time
        
        # Log performance statistics
        if self.num_processed > 0:
            logger.info(f"Processed {self.num_processed} files in {total_time:.2f} seconds")
            logger.info(f"Average processing speed: {self.num_processed / total_time:.2f} images/second")
            logger.info(f"OCR time: {self.ocr_time:.2f} seconds ({self.ocr_time / total_time * 100:.1f}%)")
            logger.info(f"LLM time: {self.llm_time:.2f} seconds ({self.llm_time / total_time * 100:.1f}%)")
        
        # Clean up temp directory if it exists
        if self.temp_dir.exists():
            try:
                shutil.rmtree(self.temp_dir)
            except Exception as e:
                logger.warning(f"Failed to clean up temp directory: {str(e)}")
    
    def process_classification_results(self, batch_results, pbar):
        """Process classification results and save files to appropriate directories"""
        for img_path, extracted_text, category, confidence in batch_results:
            try:
                # Determine destination directory
                if category == "SFW":
                    dest_dir = self.sfw_dir
                elif category == "Unsafe":
                    dest_dir = self.unsafe_dir
                else:  # NSFW
                    dest_dir = self.nsfw_dir
                
                # Copy file to destination directory
                shutil.copy2(img_path, dest_dir / Path(img_path).name)
                
                logger.debug(f"Classified {Path(img_path).name} as {category} (confidence: {confidence:.2f})")
                
                self.num_processed += 1
                pbar.update(1)
                
            except Exception as e:
                logger.error(f"Error saving {Path(img_path).name}: {str(e)}")
                pbar.update(1)

def main():
    parser = argparse.ArgumentParser(description="Screenshot Content Classifier")
    parser.add_argument("--input_dir", required=True, help="Directory containing screenshots")
    parser.add_argument("--output_dir", required=True, help="Base directory to save sorted screenshots")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for OCR processing")
    parser.add_argument("--device", choices=["cuda", "cpu"], default=None, help="Device to use")
    parser.add_argument("--sensitivity", choices=["low", "moderate", "high"], default="moderate", 
                        help="Sensitivity level for classification")
    parser.add_argument("--llm_model", type=str, default="TheBloke/Llama-3-8B-Instruct-GPTQ", 
                        help="HuggingFace model ID or path to GGUF file for LLM")
    parser.add_argument("--ocr_engine", type=str, choices=["tesseract", "easyocr", "paddle"], default="paddle",
                        help="OCR engine to use (tesseract, easyocr, or paddle)")
    parser.add_argument("--no_resize", action="store_false", dest="resize_images",
                        help="Disable image resizing before OCR (default: enabled)")
    parser.add_argument("--max_width", type=int, default=1920, help="Maximum image width for resizing")
    parser.add_argument("--max_height", type=int, default=1080, help="Maximum image height for resizing")
    parser.add_argument("--ocr_gpu_mem", type=int, default=8192, help="GPU memory allocation for PaddleOCR in MB")
    parser.add_argument("--llm_threads", type=int, default=4, help="Number of threads for LLM processing")
    parser.add_argument("--llm_batch_size", type=int, default=4, help="Batch size for LLM classification")
    parser.add_argument("--no_gpu_fallback", action="store_false", dest="ocr_gpu_fallback",
                        help="Disable fallback to CPU if GPU fails for OCR")
    
    args = parser.parse_args()
    
    classifier = ScreenshotClassifier(
        input_dir=args.input_dir,
        output_base_dir=args.output_dir,
        batch_size=args.batch_size,
        device=args.device,
        sensitivity=args.sensitivity,
        llm_model=args.llm_model,
        ocr_engine=args.ocr_engine,
        resize_images=args.resize_images,
        max_width=args.max_width,
        max_height=args.max_height,
        ocr_gpu_mem=args.ocr_gpu_mem,
        ocr_gpu_fallback=args.ocr_gpu_fallback,
        llm_threads=args.llm_threads,
        llm_batch_size=args.llm_batch_size
    )
    
    classifier.process_screenshots()

if __name__ == "__main__":
    main()