import streamlit as st
import torch
import joblib
import nltk
import numpy as np
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import json
from typing import List, Dict
import warnings
import PyPDF2
import io
warnings.filterwarnings('ignore')

# Download NLTK data
try:
    nltk.download('punkt', quiet=True)
except:
    pass

class ClinicalSummarizationApp:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.models = {}
        self.sentence_model = None
        
    @st.cache_resource
    def load_models(_self):
        """Load all trained models"""
        with st.spinner("Loading models... This may take a minute..."):
            try:
                # Load extractive models
                _self.models['single_extractive'] = joblib.load('./final_models/single_extractive.pkl')
                _self.models['multi_extractive'] = joblib.load('./final_models/multi_extractive.pkl')
                st.success("✓ Extractive models loaded")
                
                # Load single-document abstractive
                _self.models['single_abstractive_tokenizer'] = AutoTokenizer.from_pretrained(
                    './final_models/single_abstractive/final_model'
                )
                _self.models['single_abstractive_model'] = AutoModelForSeq2SeqLM.from_pretrained(
                    './final_models/single_abstractive/final_model',
                    dtype=torch.float32
                ).to(_self.device)
                st.success("✓ Single-document abstractive model loaded")
                
                # Load multi-document abstractive
                _self.models['multi_abstractive_tokenizer'] = AutoTokenizer.from_pretrained(
                    './final_models/multi_abstractive/final_model'
                )
                _self.models['multi_abstractive_model'] = AutoModelForSeq2SeqLM.from_pretrained(
                    './final_models/multi_abstractive/final_model',
                    dtype=torch.float32
                ).to(_self.device)
                st.success("✓ Multi-document abstractive model loaded")
                
                # Load sentence model for extractive
                _self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
                _self.sentence_model.to(_self.device)
                st.success("✓ All models loaded successfully!")
                
                return True
                
            except Exception as e:
                st.error(f"Error loading models: {str(e)}")
                return False
    
    def extract_sentences(self, text):
        """Extract sentences from text"""
        sentences = nltk.sent_tokenize(text)
        return [s.strip() for s in sentences if len(s.strip()) > 10]
    
    def extract_text_from_pdf(self, pdf_file):
        """Extract text from PDF file using PyPDF2"""
        try:
            # Reset file pointer to beginning
            pdf_file.seek(0)
            
            # Create a PDF reader object
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            
            text = ""
            total_pages = len(pdf_reader.pages)
            
            # Extract text from each page
            for page_num in range(total_pages):
                page = pdf_reader.pages[page_num]
                page_text = page.extract_text()
                
                if page_text:
                    text += page_text + "\n"
            
            # Clean up the text
            text = text.strip()
            
            if not text or len(text) < 10:
                return None
                
            return text
            
        except Exception as e:
            st.error(f"Error extracting text from PDF: {str(e)}")
            return None
    
    def single_extractive_summary(self, text, num_sentences=3):
        """Generate single-document extractive summary"""
        sentences = self.extract_sentences(text)
        if len(sentences) <= num_sentences:
            return ' '.join(sentences)
        
        # Get sentence embeddings
        embeddings = self.sentence_model.encode(sentences)
        
        # Predict importance using trained classifier
        importance_scores = self.models['single_extractive'].predict_proba(embeddings)[:, 1]
        
        # Select top sentences
        top_indices = np.argsort(importance_scores)[-num_sentences:]
        top_indices = sorted(top_indices)
        
        summary = ' '.join([sentences[i] for i in top_indices])
        return summary
    
    def multi_extractive_summary(self, documents, num_sentences=5):
        """Generate multi-document extractive summary"""
        all_sentences = []
        
        # Extract sentences from all documents
        for doc in documents:
            sentences = self.extract_sentences(doc)
            all_sentences.extend(sentences)
        
        if len(all_sentences) <= num_sentences:
            return ' '.join(all_sentences)
        
        # Get sentence embeddings
        embeddings = self.sentence_model.encode(all_sentences)
        
        # Predict importance using trained classifier
        importance_scores = self.models['multi_extractive'].predict_proba(embeddings)[:, 1]
        
        # Select top sentences while avoiding redundancy
        selected = []
        selected_embeddings = []
        
        for _ in range(num_sentences):
            if len(selected) >= len(all_sentences):
                break
                
            # Calculate MMR score for remaining sentences
            mmr_scores = []
            for i in range(len(all_sentences)):
                if i in selected:
                    mmr_scores.append(-float('inf'))
                    continue
                
                relevance = importance_scores[i]
                
                if selected_embeddings:
                    similarities = cosine_similarity([embeddings[i]], selected_embeddings).flatten()
                    diversity = 1 - max(similarities)
                else:
                    diversity = 1
                
                mmr_score = 0.7 * relevance + 0.3 * diversity
                mmr_scores.append(mmr_score)
            
            # Select best sentence
            best_idx = np.argmax(mmr_scores)
            if mmr_scores[best_idx] == -float('inf'):
                break
                
            selected.append(best_idx)
            selected_embeddings.append(embeddings[best_idx])
        
        selected = sorted(selected)
        summary = ' '.join([all_sentences[i] for i in selected])
        return summary
    
    def single_abstractive_summary(self, text, max_length=150, min_length=50):
        """Generate single-document abstractive summary"""
        tokenizer = self.models['single_abstractive_tokenizer']
        model = self.models['single_abstractive_model']
        
        # Prepare input
        input_text = f"summarize: {text}"
        inputs = tokenizer(
            input_text,
            truncation=True,
            max_length=448,
            return_tensors='pt'
        ).to(self.device)
        
        # Generate summary
        with torch.no_grad():
            outputs = model.generate(
                inputs['input_ids'],
                max_length=max_length,
                min_length=min_length,
                num_beams=4,
                length_penalty=2.0,
                early_stopping=True,
                no_repeat_ngram_size=3
            )
        
        summary = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return summary
    
    def multi_abstractive_summary(self, documents, max_length=200, min_length=50):
        """Generate multi-document abstractive summary"""
        tokenizer = self.models['multi_abstractive_tokenizer']
        model = self.models['multi_abstractive_model']
        
        # Prepare input
        combined_text = ' [SEP] '.join(documents)
        input_text = f"summarize multiple: {combined_text}"
        
        # Truncate if too long
        if len(input_text.split()) > 400:
            words = input_text.split()[:400]
            input_text = ' '.join(words)
        
        inputs = tokenizer(
            input_text,
            truncation=True,
            max_length=448,
            return_tensors='pt'
        ).to(self.device)
        
        # Generate summary
        with torch.no_grad():
            outputs = model.generate(
                inputs['input_ids'],
                max_length=max_length,
                min_length=min_length,
                num_beams=4,
                length_penalty=2.0,
                early_stopping=True,
                no_repeat_ngram_size=3
            )
        
        summary = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return summary
    
    def calculate_document_similarity(self, doc1, doc2):
        """Calculate similarity between two documents using embeddings"""
        try:
            # Get embeddings for both documents
            emb1 = self.sentence_model.encode(doc1)
            emb2 = self.sentence_model.encode(doc2)
            
            # Calculate cosine similarity
            similarity = cosine_similarity([emb1], [emb2])[0][0]
            return similarity
        except Exception as e:
            st.error(f"Error calculating similarity: {str(e)}")
            return 0.0

    def cluster_related_documents(self, documents, similarity_threshold=None):
        """Group documents based on content similarity"""
        # Use threshold from session state or default
        if similarity_threshold is None:
            similarity_threshold = getattr(st.session_state, 'similarity_threshold', 0.7)
        
        clusters = []
        used_indices = set()
        
        for i, doc1 in enumerate(documents):
            if i in used_indices:
                continue
                
            # Create a new cluster starting with current document
            cluster = [i]
            used_indices.add(i)
            
            # Find similar documents
            for j, doc2 in enumerate(documents):
                if j in used_indices or i == j:
                    continue
                
                similarity = self.calculate_document_similarity(doc1, doc2)
                if similarity >= similarity_threshold:
                    cluster.append(j)
                    used_indices.add(j)
            
            clusters.append(cluster)
        
        return clusters

    def generate_individual_and_combined_summaries(self, documents, summary_type, num_sentences=3, max_length=150, min_length=50):
        """Generate individual summaries and combined summaries for related documents"""
        results = {
            'individual_summaries': [],
            'cluster_summaries': []
        }
        
        # Generate individual summaries first
        st.subheader("📄 Individual Document Summaries")
        for i, doc in enumerate(documents):
            st.write(f"**Document {i+1}:**")
            
            individual_result = {
                'document_index': i,
                'content_preview': doc[:500] + "..." if len(doc) > 500 else doc,
                'extractive_summary': '',
                'abstractive_summary': ''
            }
            
            if summary_type in ["Extractive", "Both"]:
                extractive = self.single_extractive_summary(doc, num_sentences)
                st.info(f"**Extractive Summary:** {extractive}")
                individual_result['extractive_summary'] = extractive
            
            if summary_type in ["Abstractive", "Both"]:
                abstractive = self.single_abstractive_summary(doc, max_length, min_length)
                st.success(f"**Abstractive Summary:** {abstractive}")
                individual_result['abstractive_summary'] = abstractive
            
            results['individual_summaries'].append(individual_result)
            st.markdown("---")
        
        # Cluster related documents
        st.subheader("🔗 Document Clustering Analysis")
        with st.spinner("Analyzing document relationships..."):
            clusters = self.cluster_related_documents(documents)
        
        st.write(f"**Found {len(clusters)} document group(s):**")
        
        # Generate combined summaries for each cluster
        for cluster_idx, cluster in enumerate(clusters):
            # Create a more readable document list
            doc_list = [f"Document {i+1}" for i in cluster]
            doc_list_str = ", ".join(doc_list)
            
            st.write(f"### Group {cluster_idx + 1} ({doc_list_str})")
            
            if len(cluster) == 1:
                st.info("This group contains only one document. No combined summary needed.")
                # Add single document to results anyway
                single_doc = documents[cluster[0]]
                cluster_result = {
                    'cluster_index': cluster_idx,
                    'document_indices': cluster,
                    'average_similarity': 1.0,  # Single document has perfect similarity with itself
                    'combined_extractive': self.single_extractive_summary(single_doc, num_sentences) if summary_type in ["Extractive", "Both"] else "",
                    'combined_abstractive': self.single_abstractive_summary(single_doc, max_length, min_length) if summary_type in ["Abstractive", "Both"] else ""
                }
                results['cluster_summaries'].append(cluster_result)
                continue
            
            # Calculate average similarity within cluster
            similarities = []
            for i in range(len(cluster)):
                for j in range(i+1, len(cluster)):
                    sim = self.calculate_document_similarity(
                        documents[cluster[i]], 
                        documents[cluster[j]]
                    )
                    similarities.append(sim)
            
            avg_similarity = np.mean(similarities) if similarities else 0
            st.write(f"**Average similarity within group: {avg_similarity:.3f}**")
            
            # Combine documents in this cluster
            cluster_docs = [documents[i] for i in cluster]
            
            col1, col2 = st.columns(2)
            combined_extractive = ""
            combined_abstractive = ""
            
            if summary_type in ["Extractive", "Both"]:
                with col1:
                    st.subheader("📌 Combined Extractive Summary")
                    combined_extractive = self.multi_extractive_summary(cluster_docs, num_sentences + 2)
                    st.info(combined_extractive)
                    st.caption(f"Combined extractive summary • {len(combined_extractive.split())} words")
            
            if summary_type in ["Abstractive", "Both"]:
                with col2 if summary_type == "Both" else col1:
                    st.subheader("✨ Combined Abstractive Summary")
                    combined_abstractive = self.multi_abstractive_summary(cluster_docs, max_length + 50, min_length)
                    st.success(combined_abstractive)
                    st.caption(f"Combined abstractive summary • {len(combined_abstractive.split())} words")
            
            # Add to results
            cluster_result = {
                'cluster_index': cluster_idx,
                'document_indices': cluster,
                'average_similarity': avg_similarity,
                'combined_extractive': combined_extractive,
                'combined_abstractive': combined_abstractive
            }
            results['cluster_summaries'].append(cluster_result)
            
            st.markdown("---")
        
        return results

def main():
    st.set_page_config(
        page_title="Clinical Query Summarization System",
        page_icon="🏥",
        layout="wide"
    )
    
    st.title("🏥 Clinical Query Summarization System")
    st.markdown("**Powered by BioBart and Advanced NLP Models**")
    
    # Initialize app
    if 'app' not in st.session_state:
        st.session_state.app = ClinicalSummarizationApp()
        st.session_state.models_loaded = False
    
    # Load models
    if not st.session_state.models_loaded:
        st.session_state.models_loaded = st.session_state.app.load_models()
    
    if not st.session_state.models_loaded:
        st.error("Failed to load models. Please check that all models are in ./final_models/")
        return
    
    # Sidebar configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Document type
        doc_type = st.radio(
            "Document Type",
            ["Single Document", "Multiple Documents"],
            help="Choose between single or multi-document summarization"
        )
        
        # Summary type
        summary_type = st.radio(
            "Summary Type",
            ["Extractive", "Abstractive", "Both"],
            help="Extractive: Select important sentences\nAbstractive: Generate new text"
        )
        
        st.divider()

        num_sentences = 3 if doc_type == "Single Document" else 5
        max_length = 150 if doc_type == "Single Document" else 200
        min_length = 50
        
        # Parameters
        st.subheader("Parameters")
        
        if "Extractive" in summary_type or summary_type == "Both":
            num_sentences = st.slider(
                "Number of sentences (Extractive)",
                min_value=1,
                max_value=10,
                value=3 if doc_type == "Single Document" else 5,
                help="Number of sentences to extract"
            )
        
        if "Abstractive" in summary_type or summary_type == "Both":
            max_length = st.slider(
                "Maximum length (Abstractive)",
                min_value=50,
                max_value=300,
                value=150 if doc_type == "Single Document" else 200,
                help="Maximum length of generated summary"
            )
            
            min_length = st.slider(
                "Minimum length (Abstractive)",
                min_value=20,
                max_value=100,
                value=50,
                help="Minimum length of generated summary"
            )

        # Similarity threshold for document clustering (only for multiple documents)
        if doc_type == "Multiple Documents":
            similarity_threshold = st.slider(
                "Document similarity threshold",
                min_value=0.1,
                max_value=1.0,
                value=0.7,
                help="Higher values require documents to be more similar to be grouped together"
            )
            st.session_state.similarity_threshold = similarity_threshold
    
    # Main content area
    tab1, tab2, tab3 = st.tabs(["📝 Text Input", "📁 File Upload", "📚 Batch Processing"])
    
    with tab1:
        st.header("Text Input")
        
        if doc_type == "Single Document":
            text_input = st.text_area(
                "Enter your clinical text:",
                height=300,
                placeholder="Paste your clinical text here...",
                help="Enter the medical document or clinical notes to summarize"
            )
            
            if st.button("Generate Summary", type="primary"):
                if text_input:
                    with st.spinner("Generating summary..."):
                        col1, col2 = st.columns(2)
                        
                        if summary_type in ["Extractive", "Both"]:
                            with col1:
                                st.subheader("📌 Extractive Summary")
                                extractive = st.session_state.app.single_extractive_summary(
                                    text_input, num_sentences
                                )
                                st.info(extractive)
                                
                                # Word count
                                st.caption(f"Words: {len(extractive.split())}")
                        
                        if summary_type in ["Abstractive", "Both"]:
                            with col2 if summary_type == "Both" else col1:
                                st.subheader("✨ Abstractive Summary")
                                abstractive = st.session_state.app.single_abstractive_summary(
                                    text_input, max_length, min_length
                                )
                                st.success(abstractive)
                                
                                # Word count
                                st.caption(f"Words: {len(abstractive.split())}")
                else:
                    st.warning("Please enter some text to summarize")
        
        else:  # Multiple Documents
            num_docs = st.number_input("Number of documents:", min_value=2, max_value=5, value=2)
            
            documents = []
            for i in range(num_docs):
                doc = st.text_area(
                    f"Document {i+1}:",
                    height=150,
                    key=f"doc_{i}",
                    placeholder=f"Paste document {i+1} here..."
                )
                documents.append(doc)
            
            if st.button("Generate Summary", type="primary"):
                if all(documents):
                    with st.spinner("Generating summary..."):
                        col1, col2 = st.columns(2)
                        
                        if summary_type in ["Extractive", "Both"]:
                            with col1:
                                st.subheader("📌 Extractive Summary")
                                extractive = st.session_state.app.multi_extractive_summary(
                                    documents, num_sentences
                                )
                                st.info(extractive)
                                
                                # Word count
                                st.caption(f"Words: {len(extractive.split())}")
                        
                        if summary_type in ["Abstractive", "Both"]:
                            with col2 if summary_type == "Both" else col1:
                                st.subheader("✨ Abstractive Summary")
                                abstractive = st.session_state.app.multi_abstractive_summary(
                                    documents, max_length, min_length
                                )
                                st.success(abstractive)
                                
                                # Word count
                                st.caption(f"Words: {len(abstractive.split())}")
                else:
                    st.warning("Please fill in all document fields")
    
    with tab2:
        st.header("File Upload")
        
        # Determine if we should allow multiple files
        if doc_type == "Multiple Documents":
            uploaded_files = st.file_uploader(
                "Upload multiple text, JSON, or PDF files",
                type=['txt', 'json', 'pdf'],
                help="Upload multiple clinical documents for multi-document summarization",
                accept_multiple_files=True
            )
            
            if uploaded_files:
                documents_content = []
                successful_files = 0

                for uploaded_file in uploaded_files:
                    content = process_uploaded_file(uploaded_file)
                    if content:
                        documents_content.append(content)
                        successful_files += 1
                    else:
                        st.warning(f"Could not process file: {uploaded_file.name}")
                    
                if successful_files > 0:
                    st.success(f"Sucessfully processed {successful_files} out of {len(uploaded_files)} files")
                
                    if st.button("Generate Multi-Document Summary", type="primary"):
                        if len(documents_content) >= 2:
                            process_multiple_documents_content(documents_content, summary_type, num_sentences, max_length, min_length)
                        else:
                            st.error("Need at least 2 documents for multi-document summarization")
                else:
                    st.error("No files were successfully processed")
                    
        else:  # Single Document
            uploaded_file = st.file_uploader(
                "Upload a text file (.txt), JSON file (.json), or PDF file (.pdf)",
                type=['txt', 'json', 'pdf'],
                help="Upload a clinical document or structured medical data",
                accept_multiple_files=False
            )
            
            if uploaded_file:
                file_type = uploaded_file.type
                file_name = uploaded_file.name
                
                # Handle different file types
                if file_type == 'text/plain':
                    content = uploaded_file.read().decode('utf-8')
                    file_preview_title = "Text File Content"
                elif file_type == 'application/pdf':
                    uploaded_file.seek(0)
                    with st.spinner("Extracting text from PDF..."):
                        content = st.session_state.app.extract_text_from_pdf(uploaded_file)
                    
                    if content is None or len(content.strip()) < 10:
                        st.error("Failed to extract text from PDF. The PDF might be scanned or image-based.")
                        return
                    
                    file_preview_title = "PDF Extracted Text"
                    
                elif file_type == 'application/json':
                    content = uploaded_file.read().decode('utf-8')
                    file_preview_title = "JSON File Content"
                    # Parse JSON for preview
                    try:
                        json_data = json.loads(content)
                        content = json.dumps(json_data, indent=2)  # Pretty print for preview
                    except:
                        pass  # Keep original content if not valid JSON
                
                else:
                    st.error(f"Unsupported file type: {file_type}")
                    return

                # Show preview of content (truncated for large files)
                with st.expander(f"📄 View uploaded content ({file_name})"):
                    if len(content) > 10000:
                        st.warning(f"Content truncated (original: {len(content)} characters)")
                        preview_content = content[:10000] + "\n\n... [CONTENT TRUNCATED] ..."
                    else:
                        preview_content = content
                    
                    st.text_area("File content:", preview_content, height=300, disabled=True, key="file_preview")
                
                # Show file stats
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Words", len(content.split()))
                with col2:
                    st.metric("Characters", len(content))
                with col3:
                    st.metric("Sentences", len(nltk.sent_tokenize(content)))
                
                # Handle JSON files differently
                if file_type == 'application/json':
                    try:
                        # Re-read the file for processing (since we used it for preview)
                        uploaded_file.seek(0)
                        json_content = uploaded_file.read().decode('utf-8')
                        json_data = json.loads(json_content)
                        
                        if isinstance(json_data, dict) and any('answers' in v for v in json_data.values() if isinstance(v, dict)):
                            # Handle your specific data format
                            st.success(f"📊 Loaded medical Q&A dataset with {len(json_data)} entries")
                            
                            # Let user select an entry
                            entry_options = list(json_data.keys())[:20]  # Limit to first 20 entries
                            entry_id = st.selectbox("Select an entry to summarize:", entry_options)
                            
                            if entry_id and st.button("Summarize Selected Entry", key="json_summarize"):
                                entry = json_data[entry_id]
                                
                                # Extract text from the structure
                                text_to_summarize = ""
                                if 'answers' in entry:
                                    first_answer = list(entry['answers'].values())[0]
                                    if 'article' in first_answer:
                                        text_to_summarize = first_answer['article']
                                    else:
                                        st.error("No 'article' field found in the selected entry")
                                        return
                                else:
                                    st.error("No 'answers' field found in the selected entry")
                                    return
                                
                                process_text_content(text_to_summarize, summary_type, num_sentences, max_length, min_length)
                        
                        else:
                            st.info("📋 JSON content loaded")
                            # Allow user to process the entire JSON content or specific parts
                            if st.button("Process JSON Content", key="process_json"):
                                if isinstance(json_data, dict):
                                    # Convert dict to string representation
                                    text_to_summarize = str(json_data)
                                else:
                                    text_to_summarize = json_content
                                process_text_content(text_to_summarize, summary_type, num_sentences, max_length, min_length)
                    
                    except json.JSONDecodeError:
                        st.error("Invalid JSON file")
                        return
                else:
                    # For TXT and PDF files
                    if st.button("Summarize File", type="primary", key="summarize_file"):
                        process_text_content(content, summary_type, num_sentences, max_length, min_length)

    with tab3:
        # Batch Processing Section
        st.header("📚 Batch Processing")
        
        batch_file = st.file_uploader(
            "Upload JSON file for batch processing",
            type=['json'],
            key="batch_upload",
            help="JSON should contain a list of documents or a structured format"
        )
        
        if batch_file:
            try:
                data = json.loads(batch_file.read())
                
                # Handle different JSON structures
                if isinstance(data, list):
                    st.success(f"📦 Loaded {len(data)} items for processing")
                elif isinstance(data, dict):
                    data = list(data.values())
                    st.success(f"📦 Loaded {len(data)} items for processing")
                else:
                    st.error("Invalid JSON structure for batch processing")
                    return
                
                # Processing options
                col1, col2 = st.columns(2)
                with col1:
                    process_type = st.selectbox(
                        "Processing Type",
                        ["Single-Doc Extractive", "Single-Doc Abstractive",
                         "Multi-Doc Extractive", "Multi-Doc Abstractive"]
                    )
                
                with col2:
                    max_items = st.number_input(
                        "Number of items to process",
                        min_value=1,
                        max_value=len(data),
                        value=min(10, len(data))
                    )
                
                if st.button("Process Batch", type="primary"):
                    results = []
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    for i, item in enumerate(data[:max_items]):
                        progress_bar.progress((i + 1) / max_items)
                        status_text.text(f"Processing item {i+1}/{max_items}...")
                        
                        # Extract text based on structure
                        if isinstance(item, str):
                            text = item
                        elif isinstance(item, dict):
                            # Try different field names
                            text = item.get('text', item.get('article', item.get('content', str(item))))
                        else:
                            text = str(item)
                        
                        # Skip if text is too short
                        if len(text.strip()) < 10:
                            continue
                        
                        # Generate summary based on selected type
                        try:
                            if process_type == "Single-Doc Extractive":
                                summary = st.session_state.app.single_extractive_summary(text, 3)
                            elif process_type == "Single-Doc Abstractive":
                                summary = st.session_state.app.single_abstractive_summary(text)
                            elif process_type == "Multi-Doc Extractive":
                                summary = st.session_state.app.multi_extractive_summary([text], 5)
                            else:  # Multi-Doc Abstractive
                                summary = st.session_state.app.multi_abstractive_summary([text])
                            
                            results.append({
                                'id': i,
                                'original_length': len(text.split()),
                                'summary': summary,
                                'summary_length': len(summary.split()),
                                'compression_ratio': f"{len(summary.split()) / len(text.split()) * 100:.1f}%"
                            })
                        except Exception as e:
                            st.warning(f"Failed to process item {i}: {str(e)}")
                            continue
                    
                    progress_bar.empty()
                    status_text.empty()
                    st.success("✅ Batch processing complete!")
                    
                    # Display results
                    if results:
                        df = pd.DataFrame(results)
                        st.dataframe(df, use_container_width=True)
                        
                        # Download results
                        csv = df.to_csv(index=False)
                        st.download_button(
                            "📥 Download Results as CSV",
                            csv,
                            "batch_summaries.csv",
                            "text/csv",
                            key='download-csv'
                        )
                    else:
                        st.warning("No results were generated from the batch processing.")
                        
            except json.JSONDecodeError:
                st.error("Invalid JSON file for batch processing")
            except Exception as e:
                st.error(f"Error processing batch file: {str(e)}")

def process_text_content(content, summary_type, num_sentences, max_length, min_length):
    """Helper function to process text content and display summaries"""
    with st.spinner("Generating summary..."):
        col1, col2 = st.columns(2)
        
        if summary_type in ["Extractive", "Both"]:
            with col1:
                st.subheader("📌 Extractive Summary")
                extractive = st.session_state.app.single_extractive_summary(
                    content, num_sentences
                )
                st.info(extractive)
                st.caption(f"Extracted {num_sentences} key sentences • {len(extractive.split())} words")
        
        if summary_type in ["Abstractive", "Both"]:
            with col2 if summary_type == "Both" else col1:
                st.subheader("✨ Abstractive Summary")
                abstractive = st.session_state.app.single_abstractive_summary(
                    content, max_length, min_length
                )
                st.success(abstractive)
                st.caption(f"Generated summary • {len(abstractive.split())} words")
        
        # Compression ratio for both types
        if summary_type == "Both":
            original_words = len(content.split())
            abstractive_words = len(abstractive.split())
            if original_words > 0:
                compression = (abstractive_words / original_words) * 100
                st.info(f"📊 Compression ratio: {compression:.1f}% of original")

def process_uploaded_file(uploaded_file):
    """Process a single uploaded file and return its content"""
    file_type = uploaded_file.type
    file_name = uploaded_file.name
    
    try:
        if file_type == 'text/plain':
            content = uploaded_file.read().decode('utf-8')
        elif file_type == 'application/pdf':
            content = st.session_state.app.extract_text_from_pdf(uploaded_file)
            if content is None or len(content.strip()) < 10:
                st.error(f"Failed to extract text from PDF: {file_name}")
                return None
        elif file_type == 'application/json':
            content = uploaded_file.read().decode('utf-8')
        else:
            st.error(f"Unsupported file type: {file_type}")
            return None
            
        return content  # This line should be inside the try block
    except Exception as e:
        st.error(f"Error processing file {file_name}: {str(e)}")
        return None

def process_multiple_documents_content(documents_content, summary_type, num_sentences, max_length, min_length):
    """Process multiple documents with individual and combined summaries"""
    with st.spinner("Analyzing documents and generating summaries..."):
        # Use the new method that handles both individual and combined summaries
        results = st.session_state.app.generate_individual_and_combined_summaries(
            documents_content, 
            summary_type, 
            num_sentences, 
            max_length, 
            min_length
        )
    
    # Add download option for results
    if results['individual_summaries'] or results['cluster_summaries']:
        st.subheader("💾 Download Results")
        
        # Create a comprehensive results report
        report_content = "CLINICAL DOCUMENT SUMMARIZATION REPORT\n"
        report_content += "=" * 50 + "\n\n"
        
        # Individual summaries section
        report_content += "INDIVIDUAL DOCUMENT SUMMARIES\n"
        report_content += "-" * 30 + "\n"
        
        for i, summary in enumerate(results['individual_summaries']):
            report_content += f"\nDocument {i+1}:\n"
            if summary['extractive_summary']:
                report_content += f"Extractive Summary: {summary['extractive_summary']}\n"
            if summary['abstractive_summary']:
                report_content += f"Abstractive Summary: {summary['abstractive_summary']}\n"
            report_content += "\n"
        
        # Cluster summaries section
        if results['cluster_summaries']:
            report_content += "\nCOMBINED SUMMARIES FOR RELATED DOCUMENTS\n"
            report_content += "-" * 45 + "\n"
            
            for cluster in results['cluster_summaries']:
                report_content += f"\nGroup {cluster['cluster_index'] + 1} "
                report_content += f"(Documents: {[f'Doc{i+1}' for i in cluster['document_indices']]})\n"
                report_content += f"Similarity Score: {cluster['average_similarity']:.2f}\n"
                
                if cluster['combined_extractive']:
                    report_content += f"Combined Extractive: {cluster['combined_extractive']}\n"
                if cluster['combined_abstractive']:
                    report_content += f"Combined Abstractive: {cluster['combined_abstractive']}\n"
                report_content += "\n"
        
        # Download button
        st.download_button(
            "📥 Download Full Report",
            report_content,
            "clinical_summarization_report.txt",
            "text/plain",
            key='download-report'
        )

if __name__ == "__main__":
    main()