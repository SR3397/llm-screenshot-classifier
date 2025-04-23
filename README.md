
# Screenshot Content Classifier

> **IMPORTANT**: This project is currently an experimental proof-of-concept and is not production-ready. The LLM-based moderation approach has proven to be significantly slower than anticipated. Contributions toward optimizing performance are highly welcome!

An experimental tool for automatically classifying screenshot content into SFW (Safe For Work), Unsafe, and NSFW (Not Safe For Work) categories using OCR and LLM technologies.

## Overview

This proof-of-concept demonstrates an approach to content moderation for screenshots by:
1. Extracting text from screenshots using OCR (Optical Character Recognition)
2. Analyzing the extracted text with an LLM (Large Language Model)
3. Classifying content into appropriate categories
4. Sorting files into designated directories

While the concept shows promise, the current implementation has significant performance limitations, particularly in the LLM processing stage which can be extremely slow (processing speeds of ~1-2 images per second in testing, far from the target of 10+ images per second). This project serves primarily as a starting point for further development.

## Features

- **Multiple OCR Engines**: Choose between PaddleOCR, EasyOCR, or Tesseract OCR
- **Flexible LLM Support**: Use either GGUF models (via llama-cpp-python) or HuggingFace models
- **GPU Acceleration**: Fully leverages GPU for both OCR and LLM inference
- **High-Performance Processing**: Optimized for speed with batched processing and parallel execution
- **Sensitivity Settings**: Customize content filtering strictness
- **Detailed Logging**: Comprehensive logs with performance metrics
- **Robust Error Handling**: Automatic fallbacks for handling problematic images

## Requirements

- Python 3.8+
- CUDA-compatible GPU (recommended)
- 16GB+ RAM (8GB+ VRAM recommended for GPU acceleration)

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/screenshot-classifier.git
   cd screenshot-classifier
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install the core dependencies:
   ```bash
   pip install torch numpy pillow tqdm transformers
   ```

4. Install OCR engine dependencies (choose one or more):
   ```bash
   # For PaddleOCR (recommended for highest accuracy)
   pip install paddlepaddle-gpu paddleocr
   
   # For EasyOCR (good balance of speed and accuracy)
   pip install easyocr
   
   # For Tesseract OCR (fastest, requires external Tesseract installation)
   pip install pytesseract
   # Also download and install Tesseract from: https://github.com/UB-Mannheim/tesseract/wiki
   ```

5. Install LLM dependencies:
   ```bash
   pip install llama-cpp-python
   # For CUDA acceleration with llama-cpp
   pip install llama-cpp-python-cuda
   ```

## Model Setup

### OCR Models
OCR models are downloaded automatically by their respective libraries.

### LLM Models
Download a GGUF model for text classification:
1. Create a models directory: `mkdir -p models/llm`
2. Download a model like Llama-3.1-8B-Instruct-Q8_0.gguf from HuggingFace or other sources
3. Place it in the `models/llm` directory

## Usage

Basic usage:

```bash
python screenshot_classifier.py --input_dir "Input" --output_dir "Output" --llm_model "models/llm/Llama-3.1-8B-Instruct-Q8_0.gguf" --ocr_engine paddle
```

### Command Line Options

| Option | Description | Default |
|--------|-------------|---------|
| `--input_dir` | Directory containing screenshots | (Required) |
| `--output_dir` | Base directory to save sorted screenshots | (Required) |
| `--batch_size` | Batch size for OCR processing | 8 |
| `--device` | Device to use (cuda or cpu) | Auto-detect |
| `--sensitivity` | Sensitivity level (low, moderate, high) | moderate |
| `--llm_model` | HuggingFace model ID or path to GGUF file | TheBloke/Llama-3-8B-Instruct-GPTQ |
| `--ocr_engine` | OCR engine (tesseract, easyocr, or paddle) | paddle |
| `--no_resize` | Disable image resizing before OCR | Enabled by default |
| `--max_width` | Maximum image width for resizing | 1920 |
| `--max_height` | Maximum image height for resizing | 1080 |
| `--ocr_gpu_mem` | GPU memory allocation for PaddleOCR in MB | 8192 |
| `--llm_threads` | Number of threads for LLM processing | 4 |
| `--llm_batch_size` | Batch size for LLM classification | 4 |
| `--no_gpu_fallback` | Disable fallback to CPU if GPU fails for OCR | Enabled by default |

## Advanced Configuration Examples

### High Performance Mode (RTX 3090, 24GB VRAM)
```bash
python screenshot_classifier.py --input_dir "Input" --output_dir "Output" --llm_model "models/llm/Llama-3.1-8B-Instruct-Q8_0.gguf" --ocr_engine paddle --batch_size 16 --llm_threads 8 --llm_batch_size 8 --ocr_gpu_mem 12288
```

### Balanced Mode (RTX 3060, 12GB VRAM)
```bash
python screenshot_classifier.py --input_dir "Input" --output_dir "Output" --llm_model "models/llm/Llama-3.1-8B-Instruct-Q8_0.gguf" --ocr_engine paddle --batch_size 8 --llm_threads 4 --llm_batch_size 4 --ocr_gpu_mem 6144
```

### CPU-Only Mode
```bash
python screenshot_classifier.py --input_dir "Input" --output_dir "Output" --llm_model "models/llm/Llama-3.1-8B-Instruct-Q8_0.gguf" --ocr_engine tesseract --device cpu --batch_size 4 --llm_threads 8
```

## Performance Limitations and Optimization

**Current Limitations:**
- LLM inference is extremely slow, even with optimization
- Processing speed is far below the target of 10+ images per second
- Even with a high-end RTX 3090 and i7-13700K, performance is limited by the fundamental approach

For best possible performance with the current implementation:

1. **OCR Engine Selection**:
   - PaddleOCR: Best accuracy for complex screenshots
   - EasyOCR: Good balance of speed and accuracy
   - Tesseract: Fastest option, better for simple text

2. **Batch Size Tuning**:
   - Start with --batch_size 8 and --llm_batch_size 4
   - Increase if you have more VRAM available
   - Decrease if you encounter OOM (Out of Memory) errors

3. **Memory Allocation**:
   - Adjust --ocr_gpu_mem based on your GPU's VRAM
   - 8GB is suitable for 24GB GPUs when sharing with the LLM
   - Lower to 4GB-6GB for GPUs with 12GB-16GB VRAM

4. **Image Preprocessing**:
   - The --no_resize flag can increase accuracy but decrease speed
   - Default resizing to 1080p provides a good balance

5. **Consider Alternatives**:
   - For production use, traditional content moderation APIs may be more practical
   - If you primarily need text detection without semantic understanding, consider simpler keyword-based approaches

## Output Structure

The classifier creates three directories inside your output directory:

- `SFW/`: Safe for work content
- `Unsafe/`: Content that may be sensitive but not explicit
- `NSFW/`: Not safe for work (explicit) content

## Logging

Logs are saved to `screenshot_classifier.log` and include:
- Processing details for each image
- Performance metrics
- Error reports
- OCR and LLM timing statistics

## Contributing

Contributions are very welcome and needed! Areas that particularly need improvement:

1. **LLM Performance Optimization**: The current bottleneck is LLM inference speed. Ideas include:
   - Using smaller, faster models specifically trained for moderation
   - Exploring alternatives to LLMs for text classification (traditional ML classifiers)
   - Implementing more aggressive caching mechanisms
   - Optimizing prompt design for minimal token count

2. **Alternative Approaches**: The OCR + LLM pipeline may not be the most efficient approach. Consider:
   - Direct image classification with computer vision models
   - Hybrid approaches that combine multiple techniques
   - Pre-filtering techniques to reduce LLM usage

3. **Parallel Processing**: Enhancing the multi-threading architecture

Please feel free to submit a Pull Request with any improvements!

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR)
- [EasyOCR](https://github.com/JaidedAI/EasyOCR)
- [Tesseract OCR](https://github.com/tesseract-ocr/tesseract)
- [llama-cpp-python](https://github.com/abetlen/llama-cpp-python)
- [Transformers](https://github.com/huggingface/transformers)


