# Clinical Query Summarization System

## Project Overview
This project implements a comprehensive clinical query summarization system with four capabilities:
1. **Single-document extractive summarization** - Selecting key sentences from a single document
2. **Single-document abstractive summarization** - Generating new summary text from a single document
3. **Multi-document extractive summarization** - Selecting key sentences across multiple documents
4. **Multi-document abstractive summarization** - Generating new summary text from multiple documents

## Quick Start

### 1. Installation

```bash
# Clone the repository
git clone <your-repo-url>
cd clinical-summarization

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download NLTK data
python -c "import nltk; nltk.download('punkt')"
```

### 2. Dataset Preparation

Download the dataset and organize as follows:
```
data/
├── train.json
├── validation.json
└── test.json
```

Expected JSON format for each entry:
```json
{
  "id": "unique_id",
  "question": "clinical query text",
  "documents": ["document 1 text", "document 2 text", ...],
  "summary": "reference summary for training/evaluation"
}
```

### 3. Training Models

#### Train all models (recommended for first run):
```bash
# Train all 4 models (recommended):
python train.py \
    --train_file data/train.json \
    --val_file data/validation.json \
    --test_file data/test.json \
    --output_dir ./final_models

# Train only abstractive models:
python train.py --skip_extractive

# Train only extractive models:  
python train.py --skip_abstractive
```


### 4. Running the Streamlit Interface

```bash
streamlit run app.py
```

Then navigate to `http://localhost:8501` in your browser.

## Features

### Text Input Tab
- **Single Document Mode:** Process individual clinical documents
- **Multiple Documents Mode:** Process 2-5 documents simultaneously
- **Summary Types:** Extractive, Abstractive, or Both
- **Real-time Parameters:**
  - Number of sentences for extractive summaries
  - Min/max length for abstractive summaries
  - Similarity threshold for document clustering

### File Upload Tab  
- **Supported Formats:** PDF, TXT, JSON
- **PDF Text Extraction:** Automatic text extraction using PyPDF2
- **JSON Processing:** Handles structured medical Q&A data
- **Multi-file Upload:** Process multiple documents simultaneously
- **Document Clustering:** Automatically groups similar documents

### Batch Processing Tab
- **JSON batch processing:** Process entire datasets
- **Multiple processing types:** single/multi-doc, extractive/abstractive
- **Results export:** Download summaries as CSV
- **Progress Tracking:** Real-time progress bar

## Model Architecture

### Extractive Models
- **Sentence-BERT Embeddings**: Uses `all-MiniLM-L6-v2` for sentence importance scoring
- **Classification**: Logistic Regression classifier trained on sentence importance
- **Multi-document**: Combines sentences from multiple documents with redundancy removal

### Abstractive Models
- **BioBart:** (`GanjinZero/biobart-v2-base`) Optimized for biomedical text
- **Fine-tuning:** Custom training on clinical summarization tasks
- **Multi-document:** Uses [SEP] token to separate documents in input

### Document Clustering
- **Similarity Calculation:** Cosine similarity of document embeddings
- **Grouping:** Hierarchical clustering based on similarity threshold
- **Combined Summaries:** Generates individual and group-level summaries


## Evaluation Metrics

**ROUGE Metrics**
  - ROUGE-1: Unigram overlap
  - ROUGE-2: Bigram overlap  
  - ROUGE-L: Longest common subsequence
**Validation:** Early stopping with patience=3
**Test Evaluation:** Automatic evaluation on test set


## Citation
If you use this code, please cite:

- **BioBart:** https://huggingface.co/GanjinZero/biobart-v2-base
- **Sentence-BERT:** https://github.com/UKPLab/sentence-transformers