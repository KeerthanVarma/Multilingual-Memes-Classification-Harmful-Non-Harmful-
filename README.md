# Multilingual Memes Classification: Harmful vs Non-Harmful Detection

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Akhilesh348/Multilingual-Memes-Classification-Harmful-Non-Harmful-/blob/main/)

A comprehensive deep learning framework for classifying memes as harmful or non-harmful across multiple languages using multi-modal feature extraction and fusion techniques.

---

## Overview

This project implements a sophisticated multi-modal approach to detect harmful content in memes by combining:
- **Global image features** via CLIP embeddings
- **Local image features** via VGG19 with selective search
- **Text extraction** from meme images using OCR
- **Semantic relationships** via ConceptNet encoding

The framework supports multilingual meme analysis (Telugu and English) and leverages state-of-the-art deep learning models for robust harmful content detection.

---

## Key Features

- **Multi-Modal Feature Extraction**: Combines visual and textual information for comprehensive meme understanding
- **Multilingual Support**: Handles Telugu and English text within meme images
- **CLIP Integration**: Utilizes OpenAI's CLIP model for vision-language embeddings
- **Local Feature Extraction**: Advanced selective search with VGG19 for region-based features
- **OCR Processing**: GPU-accelerated image preprocessing with Tesseract OCR
- **Semantic Understanding**: ConceptNet integration for extracting semantic concepts and relationships
- **CUDA Support**: Optimized for GPU acceleration where available
- **Scalable Architecture**: Batch processing with multiprocessing support

---

## Project Structure

### Core Components

| File | Description |
|------|-------------|
| **Global_Image_Feature_Encoder.ipynb** | CLIP-based global image embedding generation for harmful/non-harmful classification |
| **Local_Image_Feature_Encoder.ipynb** | VGG19-based local feature extraction using selective search proposals |
| **Text_Encoder.ipynb** | Multi-language OCR and transformer-based text feature extraction (Telugu + English) |
| **ConceptNet_Encoder.ipynb** | Image captioning and semantic concept extraction using ConceptNet |
| **EMNLP_MOMENTA_All_DemoCode.py** | Complete pipeline combining all feature extractors for end-to-end classification |

### Additional Files

- **LICENSE.txt**: MIT License (Copyright © 2021 Shivam Sharma)

---

## Installation & Setup

### Prerequisites

- Python 3.7+
- CUDA 11.0+ (recommended for GPU acceleration)
- 8GB+ RAM
- GPU with 4GB+ VRAM (optional but recommended)

### Dependencies

Install all required packages:

```bash
pip install torch torchvision torchaudio
pip install transformers tqdm pillow pandas numpy
pip install scikit-learn opencv-python scikit-image
pip install pytesseract spacy sentence-transformers
pip install matplotlib jupyter
```

**Optional GPU-specific setup:**
```bash
# CUDA acceleration
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

**Additional setup for Tesseract OCR:**
- **Windows**: Download installer from https://github.com/UB-Mannheim/tesseract/wiki
- **Linux**: `sudo apt-get install tesseract-ocr`
- **macOS**: `brew install tesseract`

**spaCy model:**
```bash
python -m spacy download en_core_web_sm
```

---

## Quick Start

### Using Jupyter Notebooks

1. **Global Image Features:**
   ```
   Open Global_Image_Feature_Encoder.ipynb
   - Upload harmful and non-harmful meme ZIP files
   - Generates CLIP embeddings (768-dim vectors)
   ```

2. **Local Image Features:**
   ```
   Open Local_Image_Feature_Encoder.ipynb
   - Extracts region-of-interest (ROI) features
   - Uses selective search or sliding window proposals
   - Outputs VGG19 features (4096-dim vectors)
   ```

3. **Text Extraction:**
   ```
   Open Text_Encoder.ipynb
   - Preprocesses meme images for OCR
   - Extracts text in Telugu/English
   - Generates embeddings using transformer models
   ```

4. **Semantic Analysis:**
   ```
   Open ConceptNet_Encoder.ipynb
   - Generates image captions
   - Extracts semantic concepts using spaCy NER
   - Maps to ConceptNet for relationship extraction
   ```

### Using Python Script

```python
# Load and run the complete demo
python EMNLP_MOMENTA_All_DemoCode.py
```

**Data format expected:**
```
data_dir/
├── images/
│   ├── harmful_meme_1.jpg
│   ├── non_harmful_meme_1.jpg
│   └── ...
├── train.jsonl
├── val.jsonl
└── test.jsonl
```

JSONL format:
```json
{"id": "123", "image": "image_name.jpg", "label": "harmful", "text": "meme text"}
```

---

## Model Details

### Global Image Features (CLIP)
- **Model**: `openai/clip-vit-base-patch32`
- **Output Dimension**: 768
- **Purpose**: Capture semantic image content
- **Framework**: Transformers (Hugging Face)

### Local Image Features (VGG19)
- **Model**: VGG19 with ImageNet weights
- **Output Dimension**: 4096
- **Feature Layer**: FC2 (classifier layer 6)
- **Region Detection**: Selective Search + Sliding Window
- **Max Proposals**: 50 per image
- **Min Area**: 5000 pixels

### Text Features (Multilingual)
- **OCR Engine**: Tesseract
- **Languages**: Telugu, English
- **Preprocessing**: GPU-accelerated bilateral filtering
- **Embedding Model**: `sentence-transformers/all-MiniLM-L6-v2`
- **Output Dimension**: 384

### Semantic Features (ConceptNet)
- **Captioning Model**: `nlpconnect/vit-gpt2-image-captioning`
- **NER Model**: spaCy `en_core_web_sm`
- **Knowledge Graph**: ConceptNet API
- **Concept Extraction**: Noun phrases + Named Entities
- **Embedding Model**: Sentence Transformers

---

## Performance Metrics

The framework evaluates classification performance using:
- **Accuracy**: Overall correctness
- **Precision & Recall**: Per-class performance
- **F1-Score**: Harmonic mean of precision and recall
- **ROC-AUC**: Area under the receiver operating characteristic curve
- **Confusion Matrix**: Per-class breakdown
- **Hamming Loss**: Multi-label classification metric
- **MAE**: Mean absolute error for regression tasks

---

## Feature Fusion Pipeline

```
Input Image
    ↓
┌───────────────────────────────────────┐
│         Multi-Modal Extraction        │
├───────────────────────────────────────┤
│  ├─ Global Features (CLIP)            │
│  ├─ Local Features (VGG19)            │
│  ├─ Text Features (OCR + Transformers)│
│  └─ Semantic Features (ConceptNet)    │
├───────────────────────────────────────┤
│      Feature Fusion/Concatenation     │
├───────────────────────────────────────┤
│    Classification Model (Trained)     │
├───────────────────────────────────────┤
│  Output: [Harmful / Non-Harmful]      │
└───────────────────────────────────────┘
```

---

## Usage Examples

### Example 1: Extract CLIP Embeddings

```python
from transformers import CLIPModel, CLIPProcessor
import torch
from PIL import Image

device = "cuda" if torch.cuda.is_available() else "cpu"
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

image = Image.open("meme.jpg")
inputs = processor(images=image, return_tensors="pt").to(device)
embeddings = model.get_image_features(**inputs)
print(embeddings.shape)  # torch.Size([1, 512])
```

### Example 2: Extract Text from Meme

```python
import pytesseract
from PIL import Image

image = Image.open("meme.jpg")
text = pytesseract.image_to_string(image, lang='tel+eng')
print(text)
```

### Example 3: Extract Local Features

```python
import cv2
import torch
from torchvision.models import vgg19, VGG19_Weights

device = "cuda" if torch.cuda.is_available() else "cpu"
weights = VGG19_Weights.DEFAULT
vgg = vgg19(weights=weights).to(device).eval()

img = cv2.imread("meme.jpg")
# ... preprocessing and inference
```

---

## Research Context

This project is based on research presented at EMNLP/MOMENTA:
- **Focus**: Multilingual harmful content detection in visual memes
- **Novelty**: Multi-modal fusion of visual, textual, and semantic features
- **Scope**: Cross-cultural and cross-lingual meme analysis

---

## Configuration Options

Key configuration parameters in notebooks:

```python
# Global Image Features
CLIP_MODEL = "openai/clip-vit-base-patch32"
BATCH_SIZE = 32

# Local Image Features
MAX_PROPOSALS = 50
MIN_PROP_AREA = 5000
IOU_THRESHOLD = 0.95

# Text Extraction
OCR_LANGUAGES = "tel+eng"  # Telugu + English
USE_PREPROCESS = True
WORKERS = 2

# Semantic Features
TOP_N_CONCEPTNET = 10
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CAPTION_MODEL = "nlpconnect/vit-gpt2-image-captioning"
```

---

## GPU Acceleration

All components support CUDA acceleration:

```python
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Enable cuBLAS for faster operations
torch.backends.cuda.matmul.allow_tf32 = True
```

For multi-GPU support:
```python
if torch.cuda.device_count() > 1:
    model = torch.nn.DataParallel(model)
```

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| **CUDA out of memory** | Reduce batch size or use CPU mode |
| **Tesseract not found** | Install from https://github.com/UB-Mannheim/tesseract/wiki |
| **spaCy model missing** | Run `python -m spacy download en_core_web_sm` |
| **Selective Search not available** | Falls back to sliding window proposals |
| **PIL cannot open image** | Ensure image format is supported (JPG, PNG, etc.) |

---

## References

- **CLIP**: Radford et al., "Learning Transferable Models for Vision Tasks" (OpenAI)
- **VGG19**: Simonyan & Zisserman, "Very Deep Convolutional Networks for Large-Scale Image Recognition"
- **ConceptNet**: Speer et al., "ConceptNet at SemEval-2017 Task 2" (Semantic Web)
- **Transformers**: Wolf et al., "HuggingFace's Transformers: State-of-the-art NLP"

---

## License

This project is licensed under the **MIT License**. See [LICENSE.txt](LICENSE.txt) for details.

Copyright © 2021 Shivam Sharma

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files to deal in the Software without restriction.

---

## Contributing

Contributions are welcome! Please feel free to:
- Report bugs and issues
- Suggest new features or improvements
- Submit pull requests with enhancements

---

## Contact & Support

For questions, issues, or collaborations:
- **GitHub Issues**: Report bugs and feature requests
- **Discussions**: Share ideas and ask questions

---

## Acknowledgments

- OpenAI for CLIP model
- Hugging Face for Transformers library
- PyTorch and TorchVision teams
- ConceptNet community
- spaCy team for NLP tools

---

## Citation

If you use this project in your research, please cite:

```bibtex
@software{sharma2021multilingual_memes,
  title={Multilingual Memes Classification: Harmful vs Non-Harmful Detection},
  author={Sharma, Shivam},
  year={2021},
  url={https://github.com/Akhilesh348/Multilingual-Memes-Classification-Harmful-Non-Harmful-}
}
```

---

**Last Updated**: April 2026  
**Version**: 1.0.0
