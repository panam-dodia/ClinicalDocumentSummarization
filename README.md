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
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('averaged_perceptron_tagger'); nltk.download('wordnet')"
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
python train.py \
    --train_file data/train.json \
    --val_file data/validation.json \
    --test_file data/test.json \
    --model_type both \
    --task_type both \
    --model_name GanjinZero/biobart-v2-base \
    --output_dir ./trained_models
```

#### Train specific model combinations:
```bash
# Single-document extractive only
python train.py \
    --train_file data/train.json \
    --val_file data/validation.json \
    --model_type extractive \
    --task_type single

# Multi-document abstractive with BioBart
python train.py \
    --train_file data/train.json \
    --val_file data/validation.json \
    --model_type abstractive \
    --task_type multi \
    --model_name GanjinZero/biobart-v2-base

# With Weights & Biases tracking
python train.py \
    --train_file data/train.json \
    --val_file data/validation.json \
    --model_type both \
    --task_type both \
    --use_wandb
```

### 4. Running the Streamlit Interface

```bash
streamlit run app.py
```

Then navigate to `http://localhost:8501` in your browser.

## Features

### Interactive Demo Tab
- **Input Options**: Single or multi-document text input
- **Summary Types**: Choose between extractive, abstractive, or both
- **Real-time Generation**: Instant summarization with adjustable parameters
- **Parameter Control**: 
  - Number of sentences for extractive summaries
  - Min/max length for abstractive summaries

### Batch Processing Tab
- **File Upload**: Process entire JSON datasets
- **Batch Operations**: Summarize multiple documents at once
- **Export Results**: Download summaries as CSV
- **Progress Tracking**: Real-time progress bar

### Evaluation Tab
- **Comprehensive Metrics**:
  - ROUGE scores (ROUGE-1, ROUGE-2, ROUGE-L)
  - BERTScore (Precision, Recall, F1)
  - BLEU scores (1-4 gram)
  - METEOR score
  - Semantic similarity
  - Readability metrics
  - Factual consistency
- **Side-by-side Comparison**: Compare reference and generated summaries
- **Detailed Analysis**: Export full evaluation report

## Model Options

### Extractive Models
- **Sentence-BERT Embeddings**: Uses `all-MiniLM-L6-v2` for sentence importance scoring
- **Clustering (Multi-doc)**: KMeans clustering for diverse sentence selection
- **MMR Algorithm**: Maximal Marginal Relevance for balancing relevance and diversity

### Abstractive Models
- **BioBart** (`GanjinZero/biobart-v2-base`): Optimized for biomedical text
- **T5** (`t5-small` or `t5-base`): General-purpose text-to-text transformer
- **BART** (`facebook/bart-base`): Strong general summarization performance

## Advanced Features

### Medical Text Processing (`utils_evaluation.py`)
- **PHI Removal**: Automatic detection and masking of dates, SSNs, IDs
- **Entity Extraction**: Identifies medications, symptoms, procedures, measurements
- **Section Segmentation**: Parses clinical notes into standard sections
- **OCR Error Correction**: Fixes common scanning errors in medical text

### Optimization Techniques
- **MMR (Maximal Marginal Relevance)**: Balances relevance and diversity
- **Redundancy Removal**: Eliminates duplicate information
- **Query-Focused Extraction**: Prioritizes query-relevant content

## Evaluation Metrics

Following MediQA2021 guidelines (Section 4):

1. **ROUGE Metrics**
   - ROUGE-1: Unigram overlap
   - ROUGE-2: Bigram overlap  
   - ROUGE-L: Longest common subsequence
   - ROUGE-Lsum: Summary-level LCS

2. **Semantic Metrics**
   - BERTScore: Contextual embedding similarity
   - Semantic Similarity: Sentence-BERT cosine similarity

3. **Language Quality**
   - BLEU: N-gram precision
   - METEOR: Synonym-aware matching
   - Readability: Flesch scores, grade level

4. **Clinical Accuracy**
   - Factual Consistency: Entity preservation
   - Length Ratios: Compression statistics

## Project Structure

```
clinical-summarization/
├── app.py                    # Streamlit web interface
├── train.py                  # Model training script
├── utils_evaluation.py       # Evaluation metrics & utilities
├── requirements.txt          # Python dependencies
├── README.md                # This file
├── data/                    # Dataset directory
│   ├── train.json
│   ├── validation.json
│   └── test.json
├── trained_models/          # Saved model checkpoints
│   ├── extractive_single.pkl
│   ├── extractive_multi.pkl
│   ├── abstractive_single/
│   │   └── best_model/
│   └── abstractive_multi/
│       └── best_model/
└── logs/                    # Training logs
```

## Training Tips

### For Best Results
1. **Medical Domain**: Use BioBart or SciBERT-based models
2. **Learning Rate**: Start with 1e-5 for fine-tuning
3. **Batch Size**: Use gradient accumulation if GPU memory limited
4. **Epochs**: 3-5 epochs typically sufficient
5. **Validation**: Monitor ROUGE scores during training

### Hyperparameter Tuning
```python
# In train.py, modify TrainingArguments:
training_args = TrainingArguments(
    num_train_epochs=5,          # Increase for better performance
    per_device_train_batch_size=2,  # Decrease if OOM
    gradient_accumulation_steps=4,   # Increase if batch size reduced
    learning_rate=5e-5,          # Adjust based on convergence
    warmup_steps=500,            # 10% of total steps
)
```

## Common Issues & Solutions

### Out of Memory (OOM)
```bash
# Reduce batch size
--per_device_train_batch_size 2

# Enable gradient accumulation
--gradient_accumulation_steps 4

# Use smaller model
--model_name t5-small

# Enable mixed precision
--fp16 True
```

### Poor Summarization Quality
- Increase training epochs
- Use domain-specific model (BioBart)
- Clean and preprocess data
- Adjust min/max length parameters
- Fine-tune on more data

### Slow Inference
- Use GPU: `torch.cuda.is_available()`
- Batch process documents
- Cache sentence embeddings
- Use smaller models for demo

## API Usage Example

```python
from utils_evaluation import AdvancedEvaluator, DataPreprocessor
from app import ClinicalSummarizer

# Initialize
summarizer = ClinicalSummarizer()
summarizer.load_models('biobart')

# Single document extractive
text = "Patient presented with chest pain..."
summary = summarizer.single_doc_extractive(text, num_sentences=3)

# Multi-document abstractive
documents = ["Document 1...", "Document 2..."]
summary = summarizer.multi_doc_abstractive(documents, max_length=150)

# Evaluate
evaluator = AdvancedEvaluator()
metrics = evaluator.calculate_rouge(reference, hypothesis)
```

## Citation

If you use this code, please cite:
- MediQA 2021 Challenge
- BioBart paper (if using BioBart)
- Relevant evaluation metrics papers

## License

This project is for educational purposes as part of the clinical query summarization assignment.

## Contact

For questions or issues, please open a GitHub issue or contact the project maintainers.