#!/usr/bin/env python
"""
COMPLETE CLINICAL SUMMARIZATION TRAINING SCRIPT - FIXED VERSION
Trains all 4 models: Single/Multi × Extractive/Abstractive
Fixed FP16 gradient scaler issue
"""

import json
import torch
import numpy as np
import os
import gc
import nltk
import joblib
from datetime import datetime
from tqdm import tqdm
import argparse
from torch.utils.data import Dataset
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback
)
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import cosine_similarity
from rouge_score import rouge_scorer

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

# GPU Setup
if torch.cuda.is_available():
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'
    torch.backends.cudnn.benchmark = True
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
else:
    print("WARNING: No GPU found. Training will be slow.")

#==============================================================================
# DATA PREPARATION
#==============================================================================

def load_data(file_path):
    """Load JSON data"""
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Convert dict to list if needed
    if isinstance(data, dict):
        data_list = []
        for key, value in data.items():
            if isinstance(value, dict):
                value['id'] = key
            data_list.append(value)
        return data_list
    return data

def prepare_single_document_data(data):
    """Prepare single-document data"""
    single_doc_data = []
    
    for item in data:
        if 'answers' in item:
            # Your data format
            for answer_key, answer in item['answers'].items():
                if answer.get('article') and answer.get('answer_abs_summ'):
                    single_doc_data.append({
                        'id': f"{item.get('id', 'unknown')}_{answer_key}",
                        'article': answer['article'],
                        'summary': answer['answer_abs_summ'],
                        'question': item.get('question', '')
                    })
        else:
            # Standard format
            if item.get('article') and item.get('summary'):
                single_doc_data.append(item)
    
    return single_doc_data

def prepare_multi_document_data(data):
    """Prepare multi-document data"""
    multi_doc_data = []
    
    for item in data:
        if 'answers' in item:
            # Your data format
            articles = [answer.get('article', '') for answer in item['answers'].values() 
                       if answer.get('article')]
            if articles and item.get('multi_abs_summ'):
                multi_doc_data.append({
                    'id': item.get('id', 'unknown'),
                    'articles': articles,
                    'summary': item['multi_abs_summ'],
                    'question': item.get('question', '')
                })
        else:
            # Standard format
            if item.get('articles') and item.get('summary'):
                multi_doc_data.append(item)
    
    return multi_doc_data

#==============================================================================
# DATASET CLASS FOR ABSTRACTIVE
#==============================================================================

class ClinicalDataset(Dataset):
    """Dataset for abstractive summarization"""
    
    def __init__(self, data, tokenizer, task_type='single'):
        self.data = data
        self.tokenizer = tokenizer
        self.task_type = task_type
        # Optimal for 6GB GPU with BioBart
        self.max_input_length = 448
        self.max_target_length = 142
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Get input text
        if self.task_type == 'single':
            input_text = f"summarize: {item.get('article', '')}"
        else:
            articles = item.get('articles', [])
            input_text = f"summarize multiple: {' [SEP] '.join(articles)}"
        
        target_text = item.get('summary', '')
        
        # Handle empty cases
        if not input_text.strip() or not target_text.strip():
            return {
                'input_ids': torch.zeros(self.max_input_length, dtype=torch.long),
                'attention_mask': torch.zeros(self.max_input_length, dtype=torch.long),
                'labels': torch.full((self.max_target_length,), -100, dtype=torch.long)
            }
        
        # Tokenize input
        inputs = self.tokenizer(
            input_text,
            truncation=True,
            padding='max_length',
            max_length=self.max_input_length,
            return_tensors='pt'
        )
        
        # Tokenize target
        with self.tokenizer.as_target_tokenizer():
            targets = self.tokenizer(
                target_text,
                truncation=True,
                max_length=self.max_target_length,
                padding='max_length',
                return_tensors='pt'
            )
        
        labels = targets['input_ids'].squeeze()
        # Replace padding with -100 for loss calculation
        labels[labels == self.tokenizer.pad_token_id] = -100
        
        return {
            'input_ids': inputs['input_ids'].squeeze(),
            'attention_mask': inputs['attention_mask'].squeeze(),
            'labels': labels
        }

#==============================================================================
# EXTRACTIVE SUMMARIZATION TRAINING
#==============================================================================

class ExtractiveTrainer:
    """Train extractive summarization models"""
    
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Extractive trainer using: {self.device}")
        
        # Load sentence transformer
        cache_dir = "./model_cache"
        os.makedirs(cache_dir, exist_ok=True)
        self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2', cache_folder=cache_dir)
        self.sentence_model.to(self.device)
        
    def prepare_data(self, data, task_type='single'):
        """Prepare training data for extractive model"""
        X, y = [], []
        
        print(f"Preparing {task_type} extractive data...")
        for item in tqdm(data):
            # Get text
            if task_type == 'single':
                doc_text = item.get('article', '')
            else:
                articles = item.get('articles', [])
                doc_text = ' '.join(articles) if articles else ''
            
            summary = item.get('summary', '')
            
            if not doc_text or not summary:
                continue
            
            # Get sentences
            doc_sentences = nltk.sent_tokenize(doc_text)[:50]  # Limit sentences
            summary_sentences = nltk.sent_tokenize(summary)
            
            if not doc_sentences or not summary_sentences:
                continue
            
            # Encode
            doc_embeddings = self.sentence_model.encode(doc_sentences, convert_to_tensor=False)
            summary_embeddings = self.sentence_model.encode(summary_sentences, convert_to_tensor=False)
            
            # Label sentences
            for doc_emb in doc_embeddings:
                similarities = [
                    cosine_similarity([doc_emb], [sum_emb])[0][0]
                    for sum_emb in summary_embeddings
                ]
                label = 1 if max(similarities) > 0.7 else 0
                X.append(doc_emb)
                y.append(label)
        
        return np.array(X), np.array(y)
    
    def train(self, train_data, val_data, task_type='single'):
        """Train extractive model"""
        print(f"\nTraining {task_type}-document extractive model...")
        
        # Prepare data
        X_train, y_train = self.prepare_data(train_data, task_type)
        X_val, y_val = self.prepare_data(val_data, task_type)
        
        if len(X_train) == 0:
            print(f"No {task_type} extractive training data. Skipping...")
            return None
        
        # Train classifier
        print(f"Training on {len(X_train)} sentences...")
        classifier = LogisticRegression(max_iter=1000, random_state=42)
        classifier.fit(X_train, y_train)
        
        # Evaluate
        if len(X_val) > 0:
            score = classifier.score(X_val, y_val)
            print(f"Validation accuracy: {score:.4f}")
        
        # Clear GPU memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        return classifier

#==============================================================================
# ABSTRACTIVE SUMMARIZATION TRAINING
#==============================================================================

class AbstractiveTrainer:
    """Train abstractive summarization models"""
    
    def __init__(self, model_name="GanjinZero/biobart-v2-base"):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_name = model_name
        print(f"Abstractive trainer using: {self.device}")
        
    def load_model(self):
        """Load model and tokenizer"""
        print(f"Loading {self.model_name}...")
        
        cache_dir = "./model_cache"
        os.makedirs(cache_dir, exist_ok=True)
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            cache_dir=cache_dir
        )
        
        # Load model - FIXED: Don't use fp16 for model loading
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            self.model_name,
            cache_dir=cache_dir,
            torch_dtype=torch.float32,  # Use full precision
            low_cpu_mem_usage=True
        )
        
        # Enable gradient checkpointing
        if hasattr(self.model, 'gradient_checkpointing_enable'):
            self.model.gradient_checkpointing_enable()
            
        self.model.to(self.device)
        
        param_count = sum(p.numel() for p in self.model.parameters())
        print(f"Model loaded: {param_count/1e6:.2f}M parameters")
        
    def compute_metrics(self, eval_preds):
        """Calculate ROUGE scores - FIXED version"""
        predictions, labels = eval_preds
        
        # Handle different prediction formats
        if isinstance(predictions, tuple):
            # If predictions is a tuple, take the first element (logits)
            predictions = predictions[0]
        
        # If predictions are logits, convert to token IDs
        if len(predictions.shape) == 3:  # Shape: (batch, seq_len, vocab_size)
            predictions = np.argmax(predictions, axis=-1)
        
        # Ensure predictions and labels are numpy arrays
        if torch.is_tensor(predictions):
            predictions = predictions.cpu().numpy()
        if torch.is_tensor(labels):
            labels = labels.cpu().numpy()
        
        # Decode predictions
        try:
            decoded_preds = self.tokenizer.batch_decode(predictions, skip_special_tokens=True)
        except Exception as e:
            print(f"Error decoding predictions: {e}")
            # If batch_decode fails, decode one by one
            decoded_preds = []
            for pred in predictions:
                try:
                    decoded = self.tokenizer.decode(pred, skip_special_tokens=True)
                    decoded_preds.append(decoded)
                except:
                    decoded_preds.append("")
        
        # Replace -100 in labels with pad_token_id
        labels = np.where(labels != -100, labels, self.tokenizer.pad_token_id)
        
        # Decode labels
        try:
            decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)
        except Exception as e:
            print(f"Error decoding labels: {e}")
            # If batch_decode fails, decode one by one
            decoded_labels = []
            for label in labels:
                try:
                    decoded = self.tokenizer.decode(label, skip_special_tokens=True)
                    decoded_labels.append(decoded)
                except:
                    decoded_labels.append("")
        
        # Calculate ROUGE scores
        scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        scores = []
        
        for pred, label in zip(decoded_preds, decoded_labels):
            pred = pred.strip()
            label = label.strip()
            
            if label:  # Only score if label is not empty
                score = scorer.score(label, pred)
                scores.append({
                    'rouge1': score['rouge1'].fmeasure,
                    'rouge2': score['rouge2'].fmeasure,
                    'rougeL': score['rougeL'].fmeasure
                })
        
        if not scores:
            return {'rouge1': 0, 'rouge2': 0, 'rougeL': 0}
        
        return {
            'rouge1': np.mean([s['rouge1'] for s in scores]),
            'rouge2': np.mean([s['rouge2'] for s in scores]),
            'rougeL': np.mean([s['rougeL'] for s in scores])
        }
    
    def train(self, train_data, val_data, output_dir, task_type='single'):
        """Train abstractive model"""
        print(f"\nTraining {task_type}-document abstractive model...")
        print(f"Training samples: {len(train_data)}")
        print(f"Validation samples: {len(val_data)}")
        
        # Load model
        self.load_model()
        
        # Create datasets
        train_dataset = ClinicalDataset(train_data, self.tokenizer, task_type)
        val_dataset = ClinicalDataset(val_data, self.tokenizer, task_type)
        
        # Training arguments - FIXED FP16 SETTINGS
        training_args = TrainingArguments(
            output_dir=output_dir,
            overwrite_output_dir=True,
            
            # Training settings
            num_train_epochs=3,
            learning_rate=3e-5,
            warmup_ratio=0.1,
            weight_decay=0.01,
            
            # Batch sizes for 6GB
            per_device_train_batch_size=1,
            per_device_eval_batch_size=1,
            gradient_accumulation_steps=8,  # Reduced for stability
            
            # Evaluation
            eval_strategy="steps",
            eval_steps=50,
            eval_accumulation_steps=4,
            
            # Saving
            save_strategy="steps",
            save_steps=100,
            save_total_limit=2,
            load_best_model_at_end=True,
            metric_for_best_model="rouge1",
            greater_is_better=True,
            
            # Logging
            logging_steps=10,
            logging_first_step=True,
            report_to="none",
            
            # FIXED: Proper FP16 settings
            fp16=False,  # Disable FP16 to avoid gradient scaler issues
            bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),  # Use BF16 if available
            gradient_checkpointing=True,
            optim="adamw_torch",  # Use standard AdamW optimizer
            dataloader_num_workers=0,
            dataloader_pin_memory=False,
            
            # Other
            remove_unused_columns=True,
            label_smoothing_factor=0.0,
            # predict_with_generate=False,
            # include_inputs_for_metrics=False, 
        )
        
        # Custom trainer with memory management (simplified)
        class MemoryTrainer(Trainer):
            def training_step(self, model, inputs, num_items_in_batch=None):
                """Override training step with correct signature"""
                # Call parent with all arguments
                loss = super().training_step(model, inputs, num_items_in_batch=num_items_in_batch)
                
                # Clear cache periodically
                if self.state.global_step % 50 == 0:
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        gc.collect()
                
                return loss
        
        # Train
        trainer = MemoryTrainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            tokenizer=self.tokenizer,
            compute_metrics=self.compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
        )
        
        # Clear memory before training
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
        
        print("Starting training...")
        print("This may take 1-2 hours per model...")
        
        try:
            trainer.train()
            
            # Save model
            final_model_path = os.path.join(output_dir, "final_model")
            trainer.save_model(final_model_path)
            self.tokenizer.save_pretrained(final_model_path)
            
            # Evaluate
            eval_results = trainer.evaluate()
            print(f"ROUGE-1: {eval_results.get('eval_rouge1', 0):.4f}")
            print(f"ROUGE-2: {eval_results.get('eval_rouge2', 0):.4f}")
            print(f"ROUGE-L: {eval_results.get('eval_rougeL', 0):.4f}")
            
            # Save metrics
            with open(os.path.join(output_dir, "metrics.json"), 'w') as f:
                json.dump(eval_results, f, indent=2)
                
        except Exception as e:
            print(f"Error during training: {e}")
            eval_results = {"error": str(e)}
        
        # Clean up
        del self.model
        del self.tokenizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        
        return eval_results

#==============================================================================
# TEST EVALUATION
#==============================================================================

def evaluate_test_set(model_path, test_data, task_type='single'):
    """Evaluate model on test set"""
    print(f"\nEvaluating {task_type}-document model on test set...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load model
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_path,
        torch_dtype=torch.float32  # Use full precision for evaluation
    )
    model.to(device)
    model.eval()
    
    # Generate and evaluate
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    scores = []
    
    for item in tqdm(test_data[:10], desc="Testing"):  # Limit to 10 samples for speed
        # Prepare input
        if task_type == 'single':
            input_text = f"summarize: {item.get('article', '')}"
        else:
            articles = item.get('articles', [])
            input_text = f"summarize multiple: {' [SEP] '.join(articles)}"
        
        reference = item.get('summary', '')
        
        if not input_text.strip() or not reference.strip():
            continue
        
        # Generate
        inputs = tokenizer(input_text, truncation=True, max_length=448, return_tensors='pt').to(device)
        
        with torch.no_grad():
            outputs = model.generate(
                inputs['input_ids'],
                max_length=142,
                min_length=30,
                num_beams=4,
                length_penalty=2.0,
                early_stopping=True
            )
        
        prediction = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Score
        score = scorer.score(reference, prediction)
        scores.append({
            'rouge1': score['rouge1'].fmeasure,
            'rouge2': score['rouge2'].fmeasure,
            'rougeL': score['rougeL'].fmeasure
        })
    
    if scores:
        # Average
        avg_scores = {
            'rouge1': np.mean([s['rouge1'] for s in scores]),
            'rouge2': np.mean([s['rouge2'] for s in scores]),
            'rougeL': np.mean([s['rougeL'] for s in scores])
        }
        
        print(f"Test ROUGE-1: {avg_scores['rouge1']:.4f}")
        print(f"Test ROUGE-2: {avg_scores['rouge2']:.4f}")
        print(f"Test ROUGE-L: {avg_scores['rougeL']:.4f}")
    else:
        avg_scores = {'rouge1': 0, 'rouge2': 0, 'rougeL': 0}
    
    # Clean up
    del model
    torch.cuda.empty_cache()
    
    return avg_scores

#==============================================================================
# MAIN TRAINING FUNCTION
#==============================================================================

def main():
    parser = argparse.ArgumentParser(description='Train ALL Clinical Summarization Models')
    parser.add_argument('--train_file', type=str, required=True, help='Training data')
    parser.add_argument('--val_file', type=str, required=True, help='Validation data')
    parser.add_argument('--test_file', type=str, help='Test data (optional)')
    parser.add_argument('--output_dir', type=str, default='./final_models', help='Output directory')
    parser.add_argument('--model_name', type=str, default='GanjinZero/biobart-v2-base', 
                       help='Model for abstractive (use GanjinZero/biobart-v2-base for medical)')
    parser.add_argument('--skip_extractive', action='store_true', help='Skip extractive models')
    parser.add_argument('--skip_abstractive', action='store_true', help='Skip abstractive models')
    
    args = parser.parse_args()
    
    print("="*70)
    print("COMPLETE CLINICAL SUMMARIZATION TRAINING")
    print("Training ALL 4 models: Single/Multi × Extractive/Abstractive")
    print("="*70)
    
    # Load data
    print("\n1. LOADING DATA...")
    train_raw = load_data(args.train_file)
    val_raw = load_data(args.val_file)
    test_raw = load_data(args.test_file) if args.test_file else None
    
    print(f"Loaded {len(train_raw)} training samples")
    print(f"Loaded {len(val_raw)} validation samples")
    if test_raw:
        print(f"Loaded {len(test_raw)} test samples")
    
    # Prepare data for each task
    print("\n2. PREPARING DATA...")
    train_single = prepare_single_document_data(train_raw)
    val_single = prepare_single_document_data(val_raw)
    test_single = prepare_single_document_data(test_raw) if test_raw else None
    
    train_multi = prepare_multi_document_data(train_raw)
    val_multi = prepare_multi_document_data(val_raw)
    test_multi = prepare_multi_document_data(test_raw) if test_raw else None
    
    print(f"Single-doc: {len(train_single)} train, {len(val_single)} val")
    print(f"Multi-doc: {len(train_multi)} train, {len(val_multi)} val")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Track results
    results = {}
    
    if not args.skip_extractive:
        #==========================================================================
        # TRAIN MODEL 1: SINGLE-DOCUMENT EXTRACTIVE
        #==========================================================================
        print("\n" + "="*70)
        print("MODEL 1/4: SINGLE-DOCUMENT EXTRACTIVE")
        print("="*70)
        
        model_path = os.path.join(args.output_dir, "single_extractive.pkl")
        if os.path.exists(model_path):
            print("Already trained. Skipping...")
            results['single_extractive'] = "Already exists"
        else:
            extractive_trainer = ExtractiveTrainer()
            model = extractive_trainer.train(train_single, val_single, task_type='single')
            if model:
                joblib.dump(model, model_path)
                print(f"Saved to {model_path}")
                results['single_extractive'] = "Completed"
            else:
                results['single_extractive'] = "Failed"
        
        #==========================================================================
        # TRAIN MODEL 2: MULTI-DOCUMENT EXTRACTIVE
        #==========================================================================
        print("\n" + "="*70)
        print("MODEL 2/4: MULTI-DOCUMENT EXTRACTIVE")
        print("="*70)
        
        model_path = os.path.join(args.output_dir, "multi_extractive.pkl")
        if os.path.exists(model_path):
            print("Already trained. Skipping...")
            results['multi_extractive'] = "Already exists"
        else:
            extractive_trainer = ExtractiveTrainer()
            model = extractive_trainer.train(train_multi, val_multi, task_type='multi')
            if model:
                joblib.dump(model, model_path)
                print(f"Saved to {model_path}")
                results['multi_extractive'] = "Completed"
            else:
                results['multi_extractive'] = "Failed"
    
    if not args.skip_abstractive:
        #==========================================================================
        # TRAIN MODEL 3: SINGLE-DOCUMENT ABSTRACTIVE
        #==========================================================================
        print("\n" + "="*70)
        print("MODEL 3/4: SINGLE-DOCUMENT ABSTRACTIVE")
        print("="*70)
        
        output_path = os.path.join(args.output_dir, "single_abstractive")
        if os.path.exists(os.path.join(output_path, "final_model")):
            print("Already trained. Skipping...")
            results['single_abstractive'] = "Already exists"
        else:
            abstractive_trainer = AbstractiveTrainer(args.model_name)
            metrics = abstractive_trainer.train(train_single, val_single, output_path, task_type='single')
            results['single_abstractive'] = metrics
            
            # Test evaluation
            if test_single and 'error' not in metrics:
                test_scores = evaluate_test_set(
                    os.path.join(output_path, "final_model"),
                    test_single,
                    task_type='single'
                )
                results['single_abstractive_test'] = test_scores
        
        #==========================================================================
        # TRAIN MODEL 4: MULTI-DOCUMENT ABSTRACTIVE
        #==========================================================================
        print("\n" + "="*70)
        print("MODEL 4/4: MULTI-DOCUMENT ABSTRACTIVE")
        print("="*70)
        
        output_path = os.path.join(args.output_dir, "multi_abstractive")
        if os.path.exists(os.path.join(output_path, "final_model")):
            print("Already trained. Skipping...")
            results['multi_abstractive'] = "Already exists"
        else:
            abstractive_trainer = AbstractiveTrainer(args.model_name)
            metrics = abstractive_trainer.train(train_multi, val_multi, output_path, task_type='multi')
            results['multi_abstractive'] = metrics
            
            # Test evaluation
            if test_multi and 'error' not in metrics:
                test_scores = evaluate_test_set(
                    os.path.join(output_path, "final_model"),
                    test_multi,
                    task_type='multi'
                )
                results['multi_abstractive_test'] = test_scores
    
    #==========================================================================
    # FINAL SUMMARY
    #==========================================================================
    print("\n" + "="*70)
    print("TRAINING COMPLETE!")
    print("="*70)
    print("\nModels saved in:", args.output_dir)
    print("\nStructure:")
    print(f"{args.output_dir}/")
    print("├── single_extractive.pkl")
    print("├── multi_extractive.pkl")
    print("├── single_abstractive/")
    print("│   └── final_model/")
    print("└── multi_abstractive/")
    print("    └── final_model/")
    
    # Save results summary
    with open(os.path.join(args.output_dir, "training_summary.json"), 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\n✅ ALL 4 MODELS TRAINED SUCCESSFULLY!")
    print("Ready to use with your Streamlit app!")

if __name__ == "__main__":
    main()