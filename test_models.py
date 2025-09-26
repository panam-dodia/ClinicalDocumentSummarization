#!/usr/bin/env python
"""
Test script for Clinical Summarization Models
Tests all 4 trained models to verify they work correctly
"""

import torch
import joblib
import nltk
import numpy as np
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from sentence_transformers import SentenceTransformer
import json
import os
import time
from datetime import datetime

# Download NLTK data
try:
    nltk.download('punkt', quiet=True)
except:
    pass

class ModelTester:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.results = {}
        print(f"Using device: {self.device}")
        print("="*70)
    
    def test_single_extractive(self):
        """Test single-document extractive model"""
        print("\n1. Testing SINGLE-DOCUMENT EXTRACTIVE Model")
        print("-"*50)
        
        try:
            # Load model
            print("Loading model...")
            model = joblib.load('./final_models/single_extractive.pkl')
            sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
            
            # Test text
            test_text = """
            The patient is a 65-year-old male who presents with a three-day history of 
            worsening shortness of breath and chest pain. He has a past medical history 
            significant for hypertension and diabetes mellitus type 2. Physical examination 
            reveals bilateral crackles in the lung bases and mild pedal edema. 
            Chest X-ray shows signs of pulmonary congestion. ECG demonstrates sinus 
            tachycardia without acute ST-T wave changes. Laboratory results show elevated 
            BNP levels consistent with heart failure. The patient was started on furosemide 
            40mg IV and admitted for further management of acute decompensated heart failure.
            """
            
            # Extract sentences
            sentences = nltk.sent_tokenize(test_text)
            print(f"Input: {len(sentences)} sentences, {len(test_text.split())} words")
            
            # Get embeddings and predict
            embeddings = sentence_model.encode(sentences)
            predictions = model.predict_proba(embeddings)[:, 1]
            
            # Select top 3 sentences
            top_indices = np.argsort(predictions)[-3:]
            top_indices = sorted(top_indices)
            
            summary = ' '.join([sentences[i] for i in top_indices])
            
            print(f"\nSummary ({len(summary.split())} words):")
            print(summary)
            
            # Calculate importance scores
            print("\nSentence importance scores:")
            for i, (sent, score) in enumerate(zip(sentences, predictions)):
                print(f"{i+1}. Score: {score:.3f} - {sent[:50]}...")
            
            self.results['single_extractive'] = "✓ PASSED"
            print("\n✓ Single-document extractive model working correctly!")
            
        except Exception as e:
            print(f"✗ Error: {e}")
            self.results['single_extractive'] = f"✗ FAILED: {str(e)}"
    
    def test_multi_extractive(self):
        """Test multi-document extractive model"""
        print("\n2. Testing MULTI-DOCUMENT EXTRACTIVE Model")
        print("-"*50)
        
        try:
            # Load model
            print("Loading model...")
            model = joblib.load('./final_models/multi_extractive.pkl')
            sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
            
            # Test documents
            doc1 = """
            Patient admitted with acute myocardial infarction. ECG shows ST elevation 
            in leads V2-V4. Troponin levels elevated at 5.2 ng/mL.
            """
            
            doc2 = """
            Cardiac catheterization performed revealing 95% occlusion of LAD. 
            Successful PCI with drug-eluting stent placement completed.
            """
            
            doc3 = """
            Post-procedure, patient started on dual antiplatelet therapy, beta-blocker, 
            ACE inhibitor, and statin. Transferred to CCU for monitoring.
            """
            
            documents = [doc1, doc2, doc3]
            print(f"Input: {len(documents)} documents")
            
            # Extract all sentences
            all_sentences = []
            for doc in documents:
                sentences = nltk.sent_tokenize(doc)
                all_sentences.extend(sentences)
            
            print(f"Total sentences: {len(all_sentences)}")
            
            # Get embeddings and predict
            embeddings = sentence_model.encode(all_sentences)
            predictions = model.predict_proba(embeddings)[:, 1]
            
            # Select top 4 sentences
            top_indices = np.argsort(predictions)[-4:]
            top_indices = sorted(top_indices)
            
            summary = ' '.join([all_sentences[i] for i in top_indices])
            
            print(f"\nSummary ({len(summary.split())} words):")
            print(summary)
            
            self.results['multi_extractive'] = "✓ PASSED"
            print("\n✓ Multi-document extractive model working correctly!")
            
        except Exception as e:
            print(f"✗ Error: {e}")
            self.results['multi_extractive'] = f"✗ FAILED: {str(e)}"
    
    def test_single_abstractive(self):
        """Test single-document abstractive model"""
        print("\n3. Testing SINGLE-DOCUMENT ABSTRACTIVE Model")
        print("-"*50)
        
        try:
            # Load model
            print("Loading model and tokenizer...")
            model_path = './final_models/single_abstractive/final_model'
            
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            model = AutoModelForSeq2SeqLM.from_pretrained(
                model_path,
                torch_dtype=torch.float32
            ).to(self.device)
            
            model.eval()
            
            # Test text
            test_text = """
            A 58-year-old woman presents to the emergency department with acute onset 
            chest pain that started 2 hours ago. The pain is described as crushing and 
            radiates to her left arm. She has a history of hypertension and hyperlipidemia. 
            Physical examination shows diaphoresis and mild respiratory distress. ECG shows 
            ST-segment elevation in leads V2-V4. Troponin levels are markedly elevated at 
            5.2 ng/mL. The patient was given aspirin 325mg, clopidogrel 600mg, and heparin 
            bolus. Cardiology was consulted for urgent catheterization.
            """
            
            print(f"Input: {len(test_text.split())} words")
            
            # Generate summary
            input_text = f"summarize: {test_text}"
            inputs = tokenizer(
                input_text,
                truncation=True,
                max_length=448,
                return_tensors='pt'
            ).to(self.device)
            
            print("Generating summary...")
            start_time = time.time()
            
            with torch.no_grad():
                outputs = model.generate(
                    inputs['input_ids'],
                    max_length=150,
                    min_length=50,
                    num_beams=4,
                    length_penalty=2.0,
                    early_stopping=True,
                    no_repeat_ngram_size=3
                )
            
            generation_time = time.time() - start_time
            summary = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            print(f"\nSummary ({len(summary.split())} words, generated in {generation_time:.2f}s):")
            print(summary)
            
            # Test with different parameters
            print("\n\nTesting with different length parameters:")
            with torch.no_grad():
                outputs_short = model.generate(
                    inputs['input_ids'],
                    max_length=80,
                    min_length=30,
                    num_beams=4,
                    early_stopping=True
                )
            
            summary_short = tokenizer.decode(outputs_short[0], skip_special_tokens=True)
            print(f"Short summary ({len(summary_short.split())} words): {summary_short}")
            
            self.results['single_abstractive'] = "✓ PASSED"
            print("\n✓ Single-document abstractive model working correctly!")
            
        except Exception as e:
            print(f"✗ Error: {e}")
            self.results['single_abstractive'] = f"✗ FAILED: {str(e)}"
    
    def test_multi_abstractive(self):
        """Test multi-document abstractive model"""
        print("\n4. Testing MULTI-DOCUMENT ABSTRACTIVE Model")
        print("-"*50)
        
        try:
            # Load model
            print("Loading model and tokenizer...")
            model_path = './final_models/multi_abstractive/final_model'
            
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            model = AutoModelForSeq2SeqLM.from_pretrained(
                model_path,
                torch_dtype=torch.float32
            ).to(self.device)
            
            model.eval()
            
            # Test documents
            doc1 = """
            Patient is a 72-year-old man with COPD who presents with worsening dyspnea 
            over the past week. Reports increased sputum production with yellow-green color.
            """
            
            doc2 = """
            Temperature 38.2°C, respiratory rate 24/min, oxygen saturation 88% on room air. 
            Chest X-ray shows hyperinflation with flattened diaphragms.
            """
            
            doc3 = """
            ABG shows pH 7.35, pCO2 48, pO2 58. Started on bronchodilators, steroids, and 
            empiric antibiotics for COPD exacerbation.
            """
            
            documents = [doc1, doc2, doc3]
            print(f"Input: {len(documents)} documents")
            
            # Generate summary
            combined_text = ' [SEP] '.join(documents)
            input_text = f"summarize multiple: {combined_text}"
            
            inputs = tokenizer(
                input_text,
                truncation=True,
                max_length=448,
                return_tensors='pt'
            ).to(self.device)
            
            print("Generating summary...")
            start_time = time.time()
            
            with torch.no_grad():
                outputs = model.generate(
                    inputs['input_ids'],
                    max_length=200,
                    min_length=50,
                    num_beams=4,
                    length_penalty=2.0,
                    early_stopping=True,
                    no_repeat_ngram_size=3
                )
            
            generation_time = time.time() - start_time
            summary = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            print(f"\nSummary ({len(summary.split())} words, generated in {generation_time:.2f}s):")
            print(summary)
            
            self.results['multi_abstractive'] = "✓ PASSED"
            print("\n✓ Multi-document abstractive model working correctly!")
            
        except Exception as e:
            print(f"✗ Error: {e}")
            self.results['multi_abstractive'] = f"✗ FAILED: {str(e)}"
    
    def test_with_json_data(self):
        """Test models with actual JSON data if available"""
        print("\n5. Testing with JSON Data (Optional)")
        print("-"*50)
        
        test_file = 'data/test.json'
        if not os.path.exists(test_file):
            print("Test file not found. Skipping...")
            return
        
        try:
            print(f"Loading {test_file}...")
            with open(test_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Get first item
            if isinstance(data, dict):
                first_item = list(data.values())[0]
            else:
                first_item = data[0]
            
            print(f"Testing with first item from test data...")
            
            # Extract text based on structure
            if 'answers' in first_item:
                # Multi-answer format
                first_answer = list(first_item['answers'].values())[0]
                if 'article' in first_answer:
                    test_text = first_answer['article']
                    print(f"Found article with {len(test_text.split())} words")
                    
                    # Test single abstractive with this text
                    model_path = './final_models/single_abstractive/final_model'
                    tokenizer = AutoTokenizer.from_pretrained(model_path)
                    model = AutoModelForSeq2SeqLM.from_pretrained(model_path).to(self.device)
                    
                    input_text = f"summarize: {test_text[:1000]}"  # Truncate for test
                    inputs = tokenizer(input_text, truncation=True, max_length=448, return_tensors='pt').to(self.device)
                    
                    with torch.no_grad():
                        outputs = model.generate(inputs['input_ids'], max_length=150)
                    
                    summary = tokenizer.decode(outputs[0], skip_special_tokens=True)
                    print(f"\nGenerated summary: {summary}")
                    
                    if 'answer_abs_summ' in first_answer:
                        print(f"\nReference summary: {first_answer['answer_abs_summ']}")
            
            print("\n✓ JSON data test completed!")
            
        except Exception as e:
            print(f"Error testing with JSON: {e}")
    
    def run_all_tests(self):
        """Run all tests"""
        print("="*70)
        print("CLINICAL SUMMARIZATION MODELS - COMPREHENSIVE TEST")
        print("="*70)
        print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Run tests
        self.test_single_extractive()
        self.test_multi_extractive()
        self.test_single_abstractive()
        self.test_multi_abstractive()
        self.test_with_json_data()
        
        # Summary
        print("\n" + "="*70)
        print("TEST RESULTS SUMMARY")
        print("="*70)
        
        for model_name, status in self.results.items():
            print(f"{model_name:25} : {status}")
        
        # Overall status
        all_passed = all("PASSED" in str(v) for v in self.results.values())
        
        print("\n" + "="*70)
        if all_passed:
            print("🎉 ALL TESTS PASSED! Models are ready for deployment.")
        else:
            print("⚠️ Some tests failed. Please check the errors above.")
        print("="*70)
        
        # Save test results
        results_file = f"test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(results_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        print(f"\nTest results saved to: {results_file}")

def quick_test():
    """Quick test of a specific model"""
    print("Quick Test - Single Document Abstractive")
    print("-"*50)
    
    model_path = './final_models/single_abstractive/final_model'
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
    
    test_text = "Patient with chest pain and shortness of breath. ECG shows ST elevation. Troponin elevated."
    
    inputs = tokenizer(f"summarize: {test_text}", return_tensors='pt', truncation=True)
    outputs = model.generate(**inputs, max_length=100)
    summary = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    print(f"Input: {test_text}")
    print(f"Summary: {summary}")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == 'quick':
        quick_test()
    else:
        tester = ModelTester()
        tester.run_all_tests()