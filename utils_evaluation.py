"""
Utility functions and advanced evaluation metrics for clinical summarization
"""

import re
import numpy as np
from typing import List, Dict, Tuple, Optional
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from collections import Counter
import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity
from rouge_score import rouge_scorer
from bert_score import score as bert_score
import pandas as pd
from scipy import stats

# Download required NLTK data
try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('averaged_perceptron_tagger', quiet=True)
    nltk.download('wordnet', quiet=True)
except:
    pass

class DataPreprocessor:
    """Preprocessing utilities for clinical text"""
    
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        # Medical abbreviations that shouldn't be expanded
        self.medical_abbrev = {'CT', 'MRI', 'ECG', 'EKG', 'BP', 'HR', 'RR', 'O2', 'IV', 'IM'}
    
    def clean_clinical_text(self, text: str) -> str:
        """Clean clinical text while preserving medical terminology"""
        # Remove PHI patterns (dates, IDs, etc.)
        text = re.sub(r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b', '[DATE]', text)
        text = re.sub(r'\b\d{3}-\d{2}-\d{4}\b', '[SSN]', text)
        text = re.sub(r'\b[A-Z]{2}\d{6,}\b', '[ID]', text)
        
        # Normalize whitespace
        text = ' '.join(text.split())
        
        # Fix common OCR errors in medical text
        text = text.replace('1ung', 'lung')
        text = text.replace('0xygen', 'oxygen')
        
        return text
    
    def extract_medical_entities(self, text: str) -> Dict[str, List[str]]:
        """Extract medical entities from text (simplified version)"""
        entities = {
            'medications': [],
            'symptoms': [],
            'procedures': [],
            'measurements': []
        }
        
        # Simple pattern matching for common medical terms
        med_patterns = [
            (r'\b\d+\s*mg\b', 'measurements'),
            (r'\b\d+\s*ml\b', 'measurements'),
            (r'\b\d+/\d+\b', 'measurements'),  # Blood pressure
            (r'\b(?:pain|fever|cough|nausea|vomiting|dizziness)\b', 'symptoms'),
            (r'\b(?:aspirin|ibuprofen|metformin|insulin|antibiotic)\b', 'medications'),
            (r'\b(?:surgery|biopsy|x-ray|ultrasound|CT scan|MRI)\b', 'procedures')
        ]
        
        for pattern, category in med_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            entities[category].extend(matches)
        
        return entities
    
    def segment_clinical_document(self, text: str) -> Dict[str, str]:
        """Segment clinical document into sections"""
        sections = {
            'chief_complaint': '',
            'history': '',
            'examination': '',
            'assessment': '',
            'plan': ''
        }
        
        # Common section headers in clinical notes
        section_patterns = {
            'chief_complaint': r'(?:chief complaint|cc|presenting complaint)[:\s]+(.*?)(?=\n[A-Z]|\Z)',
            'history': r'(?:history|hpi|pmh|past medical)[:\s]+(.*?)(?=\n[A-Z]|\Z)',
            'examination': r'(?:physical exam|examination|pe)[:\s]+(.*?)(?=\n[A-Z]|\Z)',
            'assessment': r'(?:assessment|impression|diagnosis)[:\s]+(.*?)(?=\n[A-Z]|\Z)',
            'plan': r'(?:plan|treatment|management)[:\s]+(.*?)(?=\n[A-Z]|\Z)'
        }
        
        for section, pattern in section_patterns.items():
            match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
            if match:
                sections[section] = match.group(1).strip()
        
        return sections

class AdvancedEvaluator:
    """Advanced evaluation metrics for summarization"""
    
    def __init__(self):
        self.rouge_scorer_instance = rouge_scorer.RougeScorer(
            ['rouge1', 'rouge2', 'rougeL', 'rougeLsum'], 
            use_stemmer=True
        )
        self.bert_model = None
        self.bert_tokenizer = None
    
    def load_bert_model(self):
        """Load BERT model for BERTScore calculation"""
        if self.bert_model is None:
            from transformers import AutoTokenizer, AutoModel
            self.bert_tokenizer = AutoTokenizer.from_pretrained('microsoft/deberta-xlarge-mnli')
            self.bert_model = AutoModel.from_pretrained('microsoft/deberta-xlarge-mnli')
    
    def calculate_rouge(self, reference: str, hypothesis: str) -> Dict[str, float]:
        """Calculate comprehensive ROUGE scores"""
        scores = self.rouge_scorer_instance.score(reference, hypothesis)
        return {
            'rouge1_precision': scores['rouge1'].precision,
            'rouge1_recall': scores['rouge1'].recall,
            'rouge1_fmeasure': scores['rouge1'].fmeasure,
            'rouge2_precision': scores['rouge2'].precision,
            'rouge2_recall': scores['rouge2'].recall,
            'rouge2_fmeasure': scores['rouge2'].fmeasure,
            'rougeL_precision': scores['rougeL'].precision,
            'rougeL_recall': scores['rougeL'].recall,
            'rougeL_fmeasure': scores['rougeL'].fmeasure,
            'rougeLsum_fmeasure': scores['rougeLsum'].fmeasure
        }
    
    def calculate_bleu(self, reference: str, hypothesis: str) -> Dict[str, float]:
        """Calculate BLEU scores"""
        from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
        
        ref_tokens = word_tokenize(reference.lower())
        hyp_tokens = word_tokenize(hypothesis.lower())
        
        smoothing = SmoothingFunction().method1
        
        scores = {
            'bleu1': sentence_bleu([ref_tokens], hyp_tokens, weights=(1, 0, 0, 0), 
                                  smoothing_function=smoothing),
            'bleu2': sentence_bleu([ref_tokens], hyp_tokens, weights=(0.5, 0.5, 0, 0),
                                  smoothing_function=smoothing),
            'bleu3': sentence_bleu([ref_tokens], hyp_tokens, weights=(0.33, 0.33, 0.33, 0),
                                  smoothing_function=smoothing),
            'bleu4': sentence_bleu([ref_tokens], hyp_tokens, weights=(0.25, 0.25, 0.25, 0.25),
                                  smoothing_function=smoothing)
        }
        
        return scores
    
    def calculate_meteor(self, reference: str, hypothesis: str) -> float:
        """Calculate METEOR score"""
        try:
            from nltk.translate.meteor_score import meteor_score
            ref_tokens = word_tokenize(reference.lower())
            hyp_tokens = word_tokenize(hypothesis.lower())
            score = meteor_score([ref_tokens], hyp_tokens)
            return score
        except ImportError:
            return 0.0
    
    def calculate_factual_consistency(self, source: str, summary: str) -> float:
        """Calculate factual consistency between source and summary"""
        # Extract entities from both
        preprocessor = DataPreprocessor()
        source_entities = preprocessor.extract_medical_entities(source)
        summary_entities = preprocessor.extract_medical_entities(summary)
        
        # Check if summary entities are present in source
        consistency_scores = []
        
        for category in source_entities:
            if not summary_entities[category]:
                continue
            
            source_set = set(s.lower() for s in source_entities[category])
            summary_set = set(s.lower() for s in summary_entities[category])
            
            if summary_set:
                overlap = len(summary_set.intersection(source_set))
                consistency = overlap / len(summary_set)
                consistency_scores.append(consistency)
        
        return np.mean(consistency_scores) if consistency_scores else 1.0
    
    def calculate_semantic_similarity(self, text1: str, text2: str) -> float:
        """Calculate semantic similarity using sentence embeddings"""
        from sentence_transformers import SentenceTransformer
        
        model = SentenceTransformer('all-MiniLM-L6-v2')
        embeddings = model.encode([text1, text2])
        similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
        
        return similarity
    
    def calculate_readability(self, text: str) -> Dict[str, float]:
        """Calculate readability metrics"""
        sentences = sent_tokenize(text)
        words = word_tokenize(text)
        
        # Basic metrics
        num_sentences = len(sentences)
        num_words = len(words)
        num_syllables = sum(self._count_syllables(word) for word in words)
        
        # Flesch Reading Ease
        if num_sentences > 0 and num_words > 0:
            flesch_score = 206.835 - 1.015 * (num_words / num_sentences) - 84.6 * (num_syllables / num_words)
        else:
            flesch_score = 0
        
        # Flesch-Kincaid Grade Level
        if num_sentences > 0 and num_words > 0:
            fk_grade = 0.39 * (num_words / num_sentences) + 11.8 * (num_syllables / num_words) - 15.59
        else:
            fk_grade = 0
        
        return {
            'flesch_reading_ease': max(0, min(100, flesch_score)),
            'flesch_kincaid_grade': max(0, fk_grade),
            'avg_sentence_length': num_words / num_sentences if num_sentences > 0 else 0,
            'avg_word_length': sum(len(word) for word in words) / num_words if num_words > 0 else 0
        }
    
    def _count_syllables(self, word: str) -> int:
        """Count syllables in a word (simplified)"""
        word = word.lower()
        vowels = "aeiouy"
        syllable_count = 0
        previous_was_vowel = False
        
        for char in word:
            is_vowel = char in vowels
            if is_vowel and not previous_was_vowel:
                syllable_count += 1
            previous_was_vowel = is_vowel
        
        if word.endswith('e'):
            syllable_count -= 1
        
        return max(1, syllable_count)
    
    def comprehensive_evaluation(self, 
                                reference: str, 
                                hypothesis: str, 
                                source: Optional[str] = None) -> pd.DataFrame:
        """Perform comprehensive evaluation with all metrics"""
        results = {}
        
        # ROUGE scores
        rouge_scores = self.calculate_rouge(reference, hypothesis)
        results.update(rouge_scores)
        
        # BLEU scores
        bleu_scores = self.calculate_bleu(reference, hypothesis)
        results.update(bleu_scores)
        
        # METEOR score
        results['meteor'] = self.calculate_meteor(reference, hypothesis)
        
        # BERTScore
        try:
            P, R, F1 = bert_score([hypothesis], [reference], lang='en', verbose=False)
            results['bertscore_precision'] = P.item()
            results['bertscore_recall'] = R.item()
            results['bertscore_f1'] = F1.item()
        except:
            pass
        
        # Semantic similarity
        results['semantic_similarity'] = self.calculate_semantic_similarity(reference, hypothesis)
        
        # Readability
        readability = self.calculate_readability(hypothesis)
        results.update({f'readability_{k}': v for k, v in readability.items()})
        
        # Length statistics
        results['length_ratio'] = len(hypothesis.split()) / len(reference.split())
        results['compression_ratio'] = len(hypothesis) / len(reference)
        
        # Factual consistency (if source provided)
        if source:
            results['factual_consistency'] = self.calculate_factual_consistency(source, hypothesis)
        
        return pd.DataFrame([results])

class SummarizationOptimizer:
    """Optimization techniques for better summarization"""
    
    @staticmethod
    def maximal_marginal_relevance(sentences: List[str], 
                                   embeddings: np.ndarray,
                                   num_sentences: int,
                                   lambda_param: float = 0.7) -> List[int]:
        """
        Select sentences using Maximal Marginal Relevance (MMR)
        Balances relevance and diversity
        """
        if len(sentences) <= num_sentences:
            return list(range(len(sentences)))
        
        # Calculate document centroid
        centroid = np.mean(embeddings, axis=0)
        
        # Calculate similarity to centroid
        relevance = cosine_similarity(embeddings, [centroid]).flatten()
        
        selected = []
        remaining = list(range(len(sentences)))
        
        # Select first sentence (highest relevance)
        first_idx = np.argmax(relevance)
        selected.append(first_idx)
        remaining.remove(first_idx)
        
        # Select remaining sentences
        while len(selected) < num_sentences and remaining:
            mmr_scores = []
            
            for idx in remaining:
                # Relevance score
                rel_score = relevance[idx]
                
                # Diversity score (max similarity to selected sentences)
                if selected:
                    selected_embeddings = embeddings[selected]
                    similarities = cosine_similarity([embeddings[idx]], selected_embeddings).flatten()
                    div_score = np.max(similarities)
                else:
                    div_score = 0
                
                # MMR score
                mmr = lambda_param * rel_score - (1 - lambda_param) * div_score
                mmr_scores.append(mmr)
            
            # Select sentence with highest MMR score
            best_idx = remaining[np.argmax(mmr_scores)]
            selected.append(best_idx)
            remaining.remove(best_idx)
        
        return sorted(selected)
    
    @staticmethod
    def redundancy_removal(sentences: List[str], threshold: float = 0.8) -> List[str]:
        """Remove redundant sentences based on similarity threshold"""
        from sentence_transformers import SentenceTransformer
        
        if len(sentences) <= 1:
            return sentences
        
        model = SentenceTransformer('all-MiniLM-L6-v2')
        embeddings = model.encode(sentences)
        
        keep_indices = [0]  # Keep first sentence
        
        for i in range(1, len(sentences)):
            # Check similarity with kept sentences
            kept_embeddings = embeddings[keep_indices]
            similarities = cosine_similarity([embeddings[i]], kept_embeddings).flatten()
            
            # Keep if not too similar to any kept sentence
            if np.max(similarities) < threshold:
                keep_indices.append(i)
        
        return [sentences[i] for i in keep_indices]
    
    @staticmethod
    def query_focused_extraction(query: str, 
                                sentences: List[str],
                                num_sentences: int) -> List[str]:
        """Extract sentences most relevant to a query"""
        from sentence_transformers import SentenceTransformer
        
        model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Encode query and sentences
        query_embedding = model.encode([query])
        sentence_embeddings = model.encode(sentences)
        
        # Calculate relevance to query
        relevance_scores = cosine_similarity(sentence_embeddings, query_embedding).flatten()
        
        # Select top sentences
        top_indices = np.argsort(relevance_scores)[-num_sentences:]
        top_indices = sorted(top_indices)
        
        return [sentences[i] for i in top_indices]

# Example usage and testing
if __name__ == "__main__":
    # Test preprocessing
    preprocessor = DataPreprocessor()
    clinical_text = """
    Patient presented on 03/15/2024 with severe chest pain. 
    BP was 140/90, HR 88. Started on aspirin 81mg daily.
    CT scan showed no abnormalities. Plan: continue monitoring.
    """
    
    cleaned = preprocessor.clean_clinical_text(clinical_text)
    entities = preprocessor.extract_medical_entities(cleaned)
    sections = preprocessor.segment_clinical_document(cleaned)
    
    print("Cleaned text:", cleaned)
    print("Entities:", entities)
    print("Sections:", sections)
    
    # Test evaluation
    evaluator = AdvancedEvaluator()
    reference = "Patient has chest pain. Blood pressure elevated. Started aspirin."
    hypothesis = "Patient presented with chest pain and high blood pressure. Aspirin treatment initiated."
    
    metrics = evaluator.comprehensive_evaluation(reference, hypothesis, clinical_text)
    print("\nEvaluation Metrics:")
    print(metrics.to_string())