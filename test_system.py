"""
Test script to verify the clinical summarization system is working correctly
Run this after installation to test all components
"""

import json
import os
import sys
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from utils_evaluation import (
    DataPreprocessor, 
    AdvancedEvaluator, 
    SummarizationOptimizer
)

def test_preprocessing():
    """Test data preprocessing functionality"""
    print("\n" + "="*50)
    print("Testing Data Preprocessing")
    print("="*50)
    
    preprocessor = DataPreprocessor()
    
    # Sample clinical text with PHI
    clinical_text = """
    Patient ID: MR123456 presented on 03/15/2024 with severe chest pain.
    SSN: 123-45-6789. BP was 140/90, HR 88 bpm, O2 sat 95%.
    Started on aspirin 81mg daily and metformin 500mg BID.
    CT scan and MRI showed no abnormalities.
    Plan: Continue monitoring, follow-up in 2 weeks.
    """
    
    # Clean text
    cleaned = preprocessor.clean_clinical_text(clinical_text)
    print("\nOriginal text:")
    print(clinical_text[:200] + "...")
    print("\nCleaned text:")
    print(cleaned[:200] + "...")
    
    # Extract entities
    entities = preprocessor.extract_medical_entities(cleaned)
    print("\nExtracted Medical Entities:")
    for category, items in entities.items():
        if items:
            print(f"  {category}: {items}")
    
    return True

def test_extractive_summarization():
    """Test extractive summarization"""
    print("\n" + "="*50)
    print("Testing Extractive Summarization")
    print("="*50)
    
    try:
        from app import ClinicalSummarizer
        import nltk
        nltk.download('punkt', quiet=True)
        
        summarizer = ClinicalSummarizer()
        
        # Only load sentence model for extractive
        from sentence_transformers import SentenceTransformer
        summarizer.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Test single document
        text = """
        The patient is a 65-year-old male who presents with a three-day history of 
        worsening shortness of breath and chest pain. He has a past medical history 
        significant for hypertension and diabetes mellitus type 2. Physical examination 
        reveals bilateral crackles in the lung bases and mild pedal edema. 
        Chest X-ray shows signs of pulmonary congestion. ECG demonstrates sinus 
        tachycardia without acute ST-T wave changes. Laboratory results show elevated 
        BNP levels consistent with heart failure. The patient was started on furosemide 
        40mg IV and admitted for further management of acute decompensated heart failure.
        """
        
        summary = summarizer.single_doc_extractive(text, num_sentences=3)
        print("\nOriginal text length:", len(text.split()), "words")
        print("\nExtractive Summary (3 sentences):")
        print(summary)
        print("\nSummary length:", len(summary.split()), "words")
        
        # Test multi-document
        doc1 = "Patient presents with fever and cough. Chest X-ray shows pneumonia."
        doc2 = "Lab results indicate elevated white blood cell count. Started on antibiotics."
        doc3 = "Follow-up visit shows improvement. Continue current treatment plan."
        
        multi_summary = summarizer.multi_doc_extractive([doc1, doc2, doc3], num_sentences=2)
        print("\nMulti-document Extractive Summary:")
        print(multi_summary)
        
        return True
        
    except Exception as e:
        print(f"Error in extractive summarization: {e}")
        return False

def test_evaluation_metrics():
    """Test evaluation metrics"""
    print("\n" + "="*50)
    print("Testing Evaluation Metrics")
    print("="*50)
    
    evaluator = AdvancedEvaluator()
    
    reference = """
    Patient has acute heart failure with shortness of breath. 
    Chest X-ray shows pulmonary congestion. Started on diuretic therapy.
    """
    
    hypothesis = """
    The patient presents with heart failure and breathing difficulty.
    Imaging reveals lung congestion. Treatment with furosemide initiated.
    """
    
    print("\nReference Summary:")
    print(reference.strip())
    print("\nGenerated Summary:")
    print(hypothesis.strip())
    
    # Calculate metrics
    print("\nCalculating evaluation metrics...")
    
    # ROUGE scores
    rouge_scores = evaluator.calculate_rouge(reference, hypothesis)
    print("\nROUGE Scores:")
    print(f"  ROUGE-1 F1: {rouge_scores['rouge1_fmeasure']:.3f}")
    print(f"  ROUGE-2 F1: {rouge_scores['rouge2_fmeasure']:.3f}")
    print(f"  ROUGE-L F1: {rouge_scores['rougeL_fmeasure']:.3f}")
    
    # BLEU scores
    bleu_scores = evaluator.calculate_bleu(reference, hypothesis)
    print("\nBLEU Scores:")
    for metric, score in bleu_scores.items():
        print(f"  {metric}: {score:.3f}")
    
    # Semantic similarity
    similarity = evaluator.calculate_semantic_similarity(reference, hypothesis)
    print(f"\nSemantic Similarity: {similarity:.3f}")
    
    # Readability
    readability = evaluator.calculate_readability(hypothesis)
    print("\nReadability Metrics:")
    print(f"  Flesch Reading Ease: {readability['flesch_reading_ease']:.1f}")
    print(f"  Average Sentence Length: {readability['avg_sentence_length']:.1f} words")
    
    return True

def test_optimization():
    """Test optimization techniques"""
    print("\n" + "="*50)
    print("Testing Optimization Techniques")
    print("="*50)
    
    optimizer = SummarizationOptimizer()
    
    sentences = [
        "Patient has chest pain and shortness of breath.",
        "The patient complains of chest discomfort and breathing difficulty.",  # Redundant
        "Blood pressure is elevated at 160/95.",
        "ECG shows normal sinus rhythm.",
        "Chest X-ray reveals no acute findings.",
        "Laboratory results are within normal limits."
    ]
    
    print("\nOriginal sentences:")
    for i, sent in enumerate(sentences, 1):
        print(f"  {i}. {sent}")
    
    # Test redundancy removal
    filtered = optimizer.redundancy_removal(sentences, threshold=0.7)
    print(f"\nAfter redundancy removal (threshold=0.7):")
    for i, sent in enumerate(filtered, 1):
        print(f"  {i}. {sent}")
    
    # Test query-focused extraction
    query = "What are the diagnostic test results?"
    relevant = optimizer.query_focused_extraction(query, sentences, num_sentences=3)
    print(f"\nQuery-focused extraction for '{query}':")
    for i, sent in enumerate(relevant, 1):
        print(f"  {i}. {sent}")
    
    return True

def create_sample_data():
    """Create sample data file for testing"""
    print("\n" + "="*50)
    print("Creating Sample Data")
    print("="*50)
    
    sample_data = [
        {
            "id": "sample_001",
            "question": "What is the diagnosis and treatment plan for this patient?",
            "documents": [
                """A 58-year-old woman presents to the emergency department with 
                acute onset chest pain that started 2 hours ago. The pain is 
                described as crushing and radiates to her left arm. She has a 
                history of hypertension and hyperlipidemia. Physical examination 
                shows diaphoresis and mild respiratory distress.""",
                
                """ECG shows ST-segment elevation in leads V2-V4. Troponin levels 
                are markedly elevated at 5.2 ng/mL. Chest X-ray is unremarkable. 
                The patient was given aspirin 325mg, clopidogrel 600mg, and 
                heparin bolus. Cardiology was consulted for urgent catheterization.""",
                
                """Cardiac catheterization revealed 95% occlusion of the LAD. 
                Successful PCI was performed with drug-eluting stent placement. 
                Post-procedure, the patient was started on dual antiplatelet therapy, 
                beta-blocker, ACE inhibitor, and statin. She was admitted to the 
                CCU for monitoring."""
            ],
            "summary": """Patient diagnosed with STEMI due to LAD occlusion. 
            Underwent successful PCI with stent placement. Started on guideline-directed 
            medical therapy including dual antiplatelet therapy, beta-blocker, 
            ACE inhibitor, and statin."""
        },
        {
            "id": "sample_002",
            "question": "What are the patient's symptoms and test results?",
            "documents": [
                """A 72-year-old man with a history of COPD presents with 
                worsening dyspnea over the past week. He reports increased 
                sputum production with yellow-green color. Temperature is 38.2°C, 
                respiratory rate 24/min, oxygen saturation 88% on room air.""",
                
                """Chest X-ray shows hyperinflation with flattened diaphragms 
                consistent with COPD, no acute infiltrates. ABG on room air shows 
                pH 7.35, pCO2 48, pO2 58. WBC count elevated at 14,000. Sputum 
                culture is pending. Started on bronchodilators, steroids, and 
                empiric antibiotics."""
            ],
            "summary": """COPD exacerbation with hypoxemia and hypercapnia. 
            Fever and elevated WBC suggest bacterial infection. Treatment includes 
            bronchodilators, steroids, and antibiotics."""
        }
    ]
    
    # Save sample data
    os.makedirs("test_data", exist_ok=True)
    with open("test_data/sample.json", "w") as f:
        json.dump(sample_data, f, indent=2)
    
    print(f"Created sample data file: test_data/sample.json")
    print(f"Number of samples: {len(sample_data)}")
    
    return "test_data/sample.json"

def test_full_pipeline(data_file):
    """Test the full pipeline with sample data"""
    print("\n" + "="*50)
    print("Testing Full Pipeline")
    print("="*50)
    
    try:
        # Load sample data
        with open(data_file, 'r') as f:
            data = json.load(f)
        
        print(f"Loaded {len(data)} samples from {data_file}")
        
        # Process first sample
        sample = data[0]
        print(f"\nProcessing sample ID: {sample['id']}")
        print(f"Question: {sample['question']}")
        
        # Initialize components
        from app import ClinicalSummarizer
        summarizer = ClinicalSummarizer()
        
        # Load models
        print("\nLoading models...")
        from sentence_transformers import SentenceTransformer
        summarizer.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Test extractive summarization
        print("\nGenerating multi-document extractive summary...")
        extractive_summary = summarizer.multi_doc_extractive(
            sample['documents'], 
            num_sentences=3
        )
        print("Extractive Summary:")
        print(extractive_summary)
        
        # Evaluate
        evaluator = AdvancedEvaluator()
        metrics = evaluator.calculate_rouge(sample['summary'], extractive_summary)
        print(f"\nROUGE-L Score: {metrics['rougeL_fmeasure']:.3f}")
        
        return True
        
    except Exception as e:
        print(f"Error in full pipeline test: {e}")
        return False

def main():
    """Run all tests"""
    print("\n" + "="*60)
    print(" CLINICAL SUMMARIZATION SYSTEM - TEST SUITE")
    print("="*60)
    
    results = {}
    
    # Test each component
    tests = [
        ("Data Preprocessing", test_preprocessing),
        ("Extractive Summarization", test_extractive_summarization),
        ("Evaluation Metrics", test_evaluation_metrics),
        ("Optimization Techniques", test_optimization),
    ]
    
    for test_name, test_func in tests:
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"\nError in {test_name}: {e}")
            results[test_name] = False
    
    # Create sample data and test full pipeline
    try:
        data_file = create_sample_data()
        results["Full Pipeline"] = test_full_pipeline(data_file)
    except Exception as e:
        print(f"\nError in full pipeline: {e}")
        results["Full Pipeline"] = False
    
    # Print summary
    print("\n" + "="*60)
    print(" TEST RESULTS SUMMARY")
    print("="*60)
    
    for test_name, passed in results.items():
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"  {test_name}: {status}")
    
    total_passed = sum(results.values())
    total_tests = len(results)
    print(f"\nTotal: {total_passed}/{total_tests} tests passed")
    
    if total_passed == total_tests:
        print("\n🎉 All tests passed! The system is working correctly.")
    else:
        print("\n⚠️  Some tests failed. Please check the error messages above.")
    
    return total_passed == total_tests

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)