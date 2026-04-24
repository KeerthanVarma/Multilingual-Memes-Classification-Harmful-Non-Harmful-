# Harmful Meme Detection in Indic Languages: Dataset Curation and Baseline Development

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Akhilesh348/Multilingual-Memes-Classification-Harmful-Non-Harmful-/blob/main/)

**Research Project** | Indian Institute of Technology Gandhinagar  
**Authors**: B Keerthan Varma, K Dinesh Siddhartha, K Hemanth, M Lakshmi Manasa, Praveen Kumar, R Bhavana, V Venkat Akhilesh Naik

A comprehensive deep learning framework for zero-shot harmful meme detection across Indic languages (Hindi, Telugu, Tamil, and Kannada) using multi-modal feature extraction and fusion techniques.

---

## Overview

Memes have become a prevalent form of communication on social media, but many contain harmful content including misinformation, stereotypes, and hate speech. This research addresses the critical need for automatic harmful meme detection in Indic languages, where limited computational resources and linguistic diversity present unique challenges.

### Key Innovation
We curate large-scale Indic meme datasets and develop baseline models for zero-shot harmful meme detection by integrating:
- **Global image features** via CLIP embeddings (vision-language understanding)
- **Local image features** via VGG19 with selective search (region-based analysis)
- **Multilingual text extraction** using IndicBERT (Indic language NLP)
- **Semantic relationships** via ConceptNet (background knowledge)

These embeddings are fused using **self-attention and cross-attention mechanisms**, followed by neural network layers for binary harmful/non-harmful classification. This multi-modal approach captures textual cues, visual semantics, and contextual interpretations critical for accurate classification.

---

## Key Features

- **Multi-Modal Feature Extraction**: Combines visual, textual, and semantic information for comprehensive meme understanding
- **Indic Language Support**: First-of-its-kind datasets for Hindi, Telugu, Tamil, and Kannada; uses IndicBERT for language-specific NLP
- **Vision-Language Integration**: CLIP embeddings for semantic image understanding + IndicBERT for text interpretation
- **Local Feature Extraction**: Selective search with VGG19 for region-of-interest (ROI) analysis; extracts entity and contextual features
- **Semantic Knowledge**: ConceptNet integration for background knowledge, concept extraction, and implicit harm reasoning
- **Attention-Based Fusion**: Self-attention and cross-attention mechanisms for multimodal embedding fusion
- **Hybrid Annotation Pipeline**: LLM-assisted reasoning combined with manual verification for dataset quality
- **Balanced Evaluation**: Macro-F1 scoring and category-wise train-test splits to address dataset imbalances
- **GPU Acceleration**: CUDA optimization and efficient batch processing for scalability
- **Zero-Shot Capabilities**: Leverages pre-trained models for classification without task-specific pre-training

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

## Dataset Overview

### Curated Indic Meme Datasets

We created the first large-scale harmful meme datasets for Indic languages:

| Language | Harmful | Non-Harmful | Total | Ratio | Source |
|----------|---------|-------------|-------|-------|--------|
| **Hindi** | 3,776 | 3,070 | 6,846 | 1.23:1 | Memotion3 + MIMIC + Web-scraped |
| **Telugu** | 965 | 1,970 | 2,935 | 1:2 | Handcrafted from scratch* |
| **Tamil** | 1,282 | 1,018 | 2,300 | 1.26:1 | TamilMemes (curated) |
| **Kannada** | 573 | 1,064 | 1,637 | 1:2 | Handcrafted from scratch* |
| **TOTAL** | **6,596** | **7,122** | **13,718** | ~0.93:1 | - |

**\* First open-source datasets for Telugu and Kannada harmful memes**

**Dataset Link**: https://iitgnacin-my.sharepoint.com/personal/23110168_iitgn_ac_in/Documents/Memes%20Dataset

### Data Characteristics
- **Train/Dev/Test Split**: 60% / 20% / 20%
- **Bias Mitigation**: Category-wise distribution to ensure proportional harmful/harmless representation
- **Identity Overlap Prevention**: Memes depicting the same person distributed across splits
- **Annotation Quality**: Hybrid LLM + manual verification pipeline

### Data Challenges
- LLM-based automatic labeling struggled with subtle harmful intent
- Limited publicly available Indic meme resources required manual curation
- Low-resolution memes were discarded due to OCR preprocessing failures

---

## Installation & Setup

### Prerequisites

- **Python**: 3.7 or higher
- **GPU**: NVIDIA GPU with 4GB+ VRAM (recommended; CPU mode available)
- **CUDA**: 11.0+ for GPU acceleration
- **RAM**: 8GB+ system memory
- **Storage**: 20GB+ for models, datasets, and outputs

### Dependencies

Install all required packages:

```bash
# Core deep learning
pip install torch torchvision torchaudio

# Transformers and language models
pip install transformers tqdm pillow pandas numpy

# Computer vision and preprocessing
pip install scikit-learn opencv-python scikit-image

# NLP and text processing
pip install pytesseract spacy sentence-transformers

# IndicBERT for Indic language support
pip install indic-nlp-library

# Visualization and notebooks
pip install matplotlib jupyter

# ConceptNet (optional, can use API)
pip install conceptnet
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

**spaCy model for NER:**
```bash
python -m spacy download en_core_web_sm
```

**IndicBERT model (Indic language BERT):**
- Automatically downloaded on first use via Hugging Face transformers
- Model: `ai4bharat/IndicBERTv2-MLM` (supports all Indic languages)

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

### Text Features (Multilingual with IndicBERT)
- **Primary Model**: `IndicBERT` (AI4Bharat)
- **Supported Languages**: Hindi, Telugu, Tamil, Kannada, English
- **OCR Engine**: Tesseract (for text extraction from images)
- **Preprocessing**: GPU-accelerated bilateral filtering
- **Fallback Embedding Model**: `sentence-transformers/all-MiniLM-L6-v2` (multilingual support)
- **Output Dimension**: 768 (IndicBERT) / 384 (Sentence Transformers)
- **Special Feature**: Captures language-specific linguistic nuances and cultural context

### Semantic Features (ConceptNet)
- **Captioning Model**: `nlpconnect/vit-gpt2-image-captioning`
- **NER Model**: spaCy `en_core_web_sm`
- **Knowledge Graph**: ConceptNet API
- **Concept Extraction**: Noun phrases + Named Entities
- **Embedding Model**: Sentence Transformers

---

## Evaluation Metrics & Results

### Primary Metrics
- **Macro-F1 Score**: Primary evaluation metric to handle class imbalance between harmful and harmless memes
- **Accuracy**: Overall classification correctness

### Ablation Study Results

**Validation Performance**:

| Language | Learning Rate | Batch Size | Macro-F1 | Accuracy |
|----------|--------------|------------|----------|----------|
| Hindi | 1e-3 | 32 | 0.41 | 0.45 |
| Hindi | 1e-5 | 32 | 0.37 | 0.41 |
| Tamil | 1e-3 | 32 | 0.32 | 0.36 |
| Tamil | 1e-5 | 32 | 0.39 | 0.43 |
| Telugu | 1e-3 | 32 | 0.35 | 0.39 |
| Telugu | 1e-5 | 32 | 0.40 | 0.44 |
| Kannada | 1e-3 | 32 | 0.35 | 0.39 |
| Kannada | 1e-5 | 32 | 0.37 | 0.43 |

**Test Performance**:

| Language | Learning Rate | Batch Size | Macro-F1 | Accuracy |
|----------|--------------|------------|----------|----------|
| Hindi | 1e-3 | 32 | 0.39 | 0.43 |
| Hindi | 1e-5 | 32 | 0.36 | 0.40 |
| Tamil | 1e-3 | 32 | 0.30 | 0.33 |
| Tamil | 1e-5 | 32 | 0.38 | 0.41 |
| Telugu | 1e-3 | 32 | 0.34 | 0.38 |
| Telugu | 1e-5 | 32 | 0.39 | 0.42 |
| Kannada | 1e-3 | 32 | 0.33 | 0.37 |
| Kannada | 1e-5 | 32 | 0.37 | 0.41 |

### Key Findings
- **Lower learning rates (1e-5) yield more stable training** and better generalization
- **Validation and test scores are closely aligned**, indicating decent generalization capability
- **Performance constrained by limited training data** (15% of full dataset due to computational limits)
- **Consistent performance across languages** despite differences in dataset size and linguistic diversity
- Performance improvements expected with full dataset training and expanded computational resources

### Additional Metrics
- **Precision & Recall**: Per-class performance analysis
- **Confusion Matrix**: Detailed per-class breakdown
- **ROC-AUC**: Area under the receiver operating characteristic curve

---

## Feature Fusion Pipeline

```
Input Meme Image
        ↓
┌──────────────────────────────────────────────────────┐
│      Multi-Modal Feature Extraction                  │
├──────────────────────────────────────────────────────┤
│  ├─ Global Visual (CLIP)         → 512-dim vector   │
│  ├─ Local Visual (VGG19 ROI)     → 4096-dim vector  │
│  ├─ Text (IndicBERT)            → 768-dim vector   │
│  └─ Semantic (ConceptNet)       → N-dim vector     │
├──────────────────────────────────────────────────────┤
│   Attention-Based Fusion Mechanisms                  │
│   ├─ Self-Attention (per modality)                  │
│   └─ Cross-Attention (inter-modal interactions)     │
├──────────────────────────────────────────────────────┤
│   Concatenated Multimodal Embedding                  │
│        ↓ (Neural Network Layers)                     │
│   Binary Classification Head                         │
├──────────────────────────────────────────────────────┤
│  Output: [Harmful / Non-Harmful] + Confidence Score │
└──────────────────────────────────────────────────────┘
```

**Architecture Parameter Count**: 3.74M parameters (excluding frozen encoders)

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

## Training Configuration

### Model Training Setup
All pre-trained encoders (IndicBERT, CLIP, VGG-19, ConceptNet) were frozen, while only final neural layers were trained:

```python
# Optimizer Configuration
optimizer = "Adam"
learning_rate = [1e-3, 1e-5]  # Tested both
epochs = 20
batch_size = 32

# Data Split
train_ratio = 0.60
dev_ratio = 0.20
test_ratio = 0.20

# Training Constraint
full_dataset_usage = 15  # percent (due to computational limits)
```

### Key Configuration Parameters

```python
# Global Image Features (CLIP)
CLIP_MODEL = "openai/clip-vit-base-patch32"
CLIP_OUTPUT_DIM = 512

# Local Image Features (VGG19)
MAX_PROPOSALS = 50  # max ROI proposals per image
MIN_PROP_AREA = 5000  # minimum proposal area in pixels
IOU_THRESHOLD = 0.95  # intersection-over-union threshold
VGG19_OUTPUT_DIM = 4096

# Text Extraction (Multilingual)
OCR_LANGUAGES = "tel+eng+hin+tam+kan"  # All Indic languages
TEXT_MODEL = "ai4bharat/IndicBERTv2-MLM"  # IndicBERT
USE_GPU_PREPROCESS = True
WORKERS = 2

# Semantic Features (ConceptNet)
TOP_N_CONCEPTNET = 10  # top N concepts extracted
CAPTION_MODEL = "nlpconnect/vit-gpt2-image-captioning"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Attention Fusion
USE_SELF_ATTENTION = True
USE_CROSS_ATTENTION = True
ATTENTION_HEADS = 8
```

### Computational Resources Used
- **GPU**: 7x T4 GPU (15 GB each) on Google Colab
- **CPU**: 7x 1 core
- **RAM**: 7x 12.7 GB RAM
- **Storage**: 20 GB out of 700 GB available

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

## Team & Contributors

**Research Team** (Indian Institute of Technology Gandhinagar):
- **B Keerthan Varma** - Dataset curation, Documentation, Report writing, Experimentation
- **K Dinesh Siddhartha** - Dataset curation, ConceptNet integration, Experimentation
- **K Hemanth** - Dataset curation, Data preprocessing, ConceptNet integration
- **M Lakshmi Manasa** - Dataset curation, Evaluation metrics, Ablation analysis
- **Praveen Kumar** - Dataset curation, Model training, Image encoder integration
- **R Bhavana** - Dataset curation, Documentation, Result summarization
- **V Venkat Akhilesh Naik** - Dataset curation, Hyperparameter tuning, IndicBERT integration

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



## References

- **MOMENTA Paper**: https://arxiv.org/pdf/2109.05184 (Multi-modal fusion architecture)
- **MIND Paper**: https://arxiv.org/pdf/2507.06908 (Zero-shot harmful content detection)
- **IndicBERT**: AI4Bharat's language model for Indic languages
- **CLIP**: Radford et al., OpenAI (Vision-Language pretraining)
- **VGG19**: Simonyan & Zisserman (Deep CNN architecture)
- **ConceptNet**: Speer et al. (Knowledge graph resource)

---

**Last Updated**: April 2026  
**Version**: 2.0.0 (Research Edition)  
**Status**: Active Research Project
