#!/usr/bin/env python3
"""
MemoryAI Enterprise - RAG Fusion System
Combines dense retrieval, sparse retrieval, graph traversal, and ColBERT for best-in-class retrieval
"""

import os
import sys
import json
import time
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import logging
from sklearn.feature_extraction.text import TfidfVectorizer
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModel
import sqlite3
from collections import defaultdict
import hashlib

# Import WPMG builder
from wpmg_builder import WPMGBuilder, get_wpmg_builder

@dataclass
class RetrievalResult:
    """Retrieval result from single method"""
    doc_id: str
    content: str
    score: float
    method: str
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class FusionResult:
    """Fused retrieval result"""
    doc_id: str
    content: str
    fusion_score: float
    individual_scores: Dict[str, float]
    metadata: Dict[str, Any] = field(default_factory=dict)

class RetrievalMethod(ABC):
    """Base class for retrieval methods"""
    
    @abstractmethod
    def retrieve(self, query: str, k: int = 10) -> List[RetrievalResult]:
        pass
    
    @abstractmethod
    def score(self, query: str, doc: str) -> float:
        pass

class DenseRetrieval(RetrievalMethod):
    """Dense retrieval using sentence embeddings"""
    
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self.embedding_cache = {}
        self.logger = logging.getLogger(__name__)
        
    def encode(self, text: str) -> np.ndarray:
        """Encode text to embedding vector"""
        text_hash = hashlib.md5(text.encode()).hexdigest()
        
        if text_hash not in self.embedding_cache:
            embedding = self.model.encode(text, convert_to_numpy=True)
            self.embedding_cache[text_hash] = embedding
        
        return self.embedding_cache[text_hash]
    
    def score(self, query: str, doc: str) -> float:
        """Compute cosine similarity between query and document"""
        query_embedding = self.encode(query)
        doc_embedding = self.encode(doc)
        
        similarity = np.dot(query_embedding, doc_embedding) / (
            np.linalg.norm(query_embedding) * np.linalg.norm(doc_embedding)
        )
        
        return float(similarity)
    
    def retrieve(self, query: str, candidates: List[Dict], k: int = 10) -> List[RetrievalResult]:
        """Retrieve top-k documents using dense embeddings"""
        
        if not candidates:
            return []
        
        # Encode query
        query_embedding = self.encode(query)
        
        # Score all candidates
        results = []
        for candidate in candidates:
            doc_embedding = self.encode(candidate['content'])
            
            similarity = np.dot(query_embedding, doc_embedding) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(doc_embedding)
            )
            
            results.append(RetrievalResult(
                doc_id=candidate['node_id'],
                content=candidate['content'],
                score=float(similarity),
                method="dense",
                metadata={'embedding_norm': float(np.linalg.norm(doc_embedding))}
            ))
        
        # Sort by score and return top-k
        results.sort(key=lambda x: x.score, reverse=True)
        return results[:k]

class SparseRetrieval(RetrievalMethod):
    """Sparse retrieval using BM25/TF-IDF"""
    
    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            max_features=10000,
            stop_words='english',
            ngram_range=(1, 2),
            use_idf=True,
            smooth_idf=True
        )
        self.doc_vectors = None
        self.logger = logging.getLogger(__name__)
    
    def fit(self, documents: List[str]):
        """Fit TF-IDF vectorizer on document collection"""
        self.doc_vectors = self.vectorizer.fit_transform(documents)
        
    def score(self, query: str, doc: str) -> float:
        """Compute BM25-like score (simplified)"""
        if self.doc_vectors is None:
            # Lazy fit if not already done
            self.fit([doc])
        
        query_vector = self.vectorizer.transform([query])
        doc_vector = self.vectorizer.transform([doc])
        
        # Cosine similarity as proxy for BM25
        similarity = cosine_similarity(query_vector, doc_vector)[0][0]
        
        return float(similarity)
    
    def retrieve(self, query: str, candidates: List[Dict], k: int = 10) -> List[RetrievalResult]:
        """Retrieve top-k documents using sparse features"""
        
        if not candidates:
            return []
        
        # Prepare documents
        documents = [c['content'] for c in candidates]
        
        # Fit vectorizer if needed
        if self.doc_vectors is None or self.doc_vectors.shape[0] != len(documents):
            self.fit(documents)
        
        # Transform query
        query_vector = self.vectorizer.transform([query])
        
        # Compute similarities
        similarities = cosine_similarity(query_vector, self.doc_vectors).flatten()
        
        # Create results
        results = []
        for i, candidate in enumerate(candidates):
            results.append(RetrievalResult(
                doc_id=candidate['node_id'],
                content=candidate['content'],
                score=float(similarities[i]),
                method="sparse",
                metadata={'tfidf_features': len(self.vectorizer.get_feature_names_out())}
            ))
        
        # Sort by score and return top-k
        results.sort(key=lambda x: x.score, reverse=True)
        return results[:k]

class GraphRetrieval(RetrievalMethod):
    """Graph-based retrieval using WPMG"""
    
    def __init__(self, user_id: str):
        self.user_id = user_id
        self.wpmg = get_wpmg_builder(user_id)
        self.logger = logging.getLogger(__name__)
    
    def score(self, query: str, doc: str) -> float:
        """Score based on graph centrality and connections"""
        # This is a simplified version
        # In practice, would use personalized PageRank or other graph algorithms
        
        # Get node from WPMG if it exists
        try:
            # Find node ID for document
            cursor = self.wpmg.conn.execute('''
                SELECT node_id FROM memory_nodes 
                WHERE content = ? LIMIT 1
            ''', (doc,))
            row = cursor.fetchone()
            
            if row:
                node_id = row[0]
                # Get centrality score
                if node_id in self.wpmg.graph:
                    centrality = nx.degree_centrality(self.wpmg.graph).get(node_id, 0)
                    return centrality
            
        except Exception as e:
            self.logger.warning(f"Graph scoring failed: {e}")
        
        return 0.0
    
    def retrieve(self, query: str, candidates: List[Dict], k: int = 10) -> List[RetrievalResult]:
        """Retrieve using graph-based features"""
        
        results = []
        
        for candidate in candidates:
            # Get node from WPMG
            cursor = self.wpmg.conn.execute('''
                SELECT node_id, importance FROM memory_nodes 
                WHERE node_id = ?
            ''', (candidate['node_id'],))
            row = cursor.fetchone()
            
            if row:
                node_id = row[0]
                importance = row[1]
                
                # Get graph context
                context = self.wpmg.get_memory_context(node_id, hop_distance=1)
                
                # Score based on importance and connectivity
                score = importance * (1 + len(context['edges']) * 0.1)
                
                results.append(RetrievalResult(
                    doc_id=candidate['node_id'],
                    content=candidate['content'],
                    score=score,
                    method="graph",
                    metadata={
                        'importance': importance,
                        'degree': len(context['edges'])
                    }
                ))
        
        # Sort by score and return top-k
        results.sort(key=lambda x: x.score, reverse=True)
        return results[:k]

class ColBERTRetrieval(RetrievalMethod):
    """ColBERT late-interaction retrieval"""
    
    def __init__(self, model_name: str = "colbert-ir/colbertv2.0"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.logger = logging.getLogger(__name__)
        
        # Move to GPU if available
        if torch.cuda.is_available():
            self.model = self.model.cuda()
    
    def encode_tokens(self, text: str) -> torch.Tensor:
        """Encode text to token-level embeddings"""
        tokens = self.tokenizer(text, return_tensors='pt', truncation=True, max_length=512)
        
        if torch.cuda.is_available():
            tokens = {k: v.cuda() for k, v in tokens.items()}
        
        with torch.no_grad():
            embeddings = self.model(**tokens).last_hidden_state
        
        return embeddings.squeeze(0)  # Remove batch dimension
    
    def score(self, query: str, doc: str) -> float:
        """Compute ColBERT late-interaction score"""
        
        query_embeddings = self.encode_tokens(query)
        doc_embeddings = self.encode_tokens(doc)
        
        # MaxSim: maximum similarity over document tokens for each query token
        scores = []
        for q_emb in query_embeddings:
            similarities = torch.cosine_similarity(q_emb.unsqueeze(0), doc_embeddings, dim=1)
            scores.append(torch.max(similarities).item())
        
        return np.mean(scores)
    
    def retrieve(self, query: str, candidates: List[Dict], k: int = 10) -> List[RetrievalResult]:
        """Retrieve using ColBERT late-interaction"""
        
        if not candidates:
            return []
        
        # Encode query
        query_embeddings = self.encode_tokens(query)
        
        results = []
        for candidate in candidates:
            try:
                doc_embeddings = self.encode_tokens(candidate['content'])
                
                # Compute ColBERT score
                scores = []
                for q_emb in query_embeddings:
                    similarities = torch.cosine_similarity(q_emb.unsqueeze(0), doc_embeddings, dim=1)
                    scores.append(torch.max(similarities).item())
                
                colbert_score = np.mean(scores)
                
                results.append(RetrievalResult(
                    doc_id=candidate['node_id'],
                    content=candidate['content'],
                    score=colbert_score,
                    method="colbert",
                    metadata={
                        'query_tokens': len(query_embeddings),
                        'doc_tokens': len(doc_embeddings)
                    }
                ))
                
            except Exception as e:
                self.logger.warning(f"ColBERT retrieval failed for candidate: {e}")
                continue
        
        # Sort by score and return top-k
        results.sort(key=lambda x: x.score, reverse=True)
        return results[:k]

class RAGFusion:
    """RAG Fusion system combining multiple retrieval methods"""
    
    def __init__(self, user_id: str):
        self.user_id = user_id
        self.retrievers = {
            'dense': DenseRetrieval(),
            'sparse': SparseRetrieval(),
            'graph': GraphRetrieval(user_id),
            'colbert': ColBERTRetrieval()
        }
        
        # Fusion weights (can be learned)
        self.fusion_weights = {
            'dense': 0.25,
            'sparse': 0.25,
            'graph': 0.25,
            'colbert': 0.25
        }
        
        self.logger = logging.getLogger(__name__)
        self.wpmg = get_wpmg_builder(user_id)
    
    def get_candidates(self, query: str, max_candidates: int = 100) -> List[Dict]:
        """Get candidate memories from WPMG"""
        
        # Query WPMG for relevant memories
        memories = self.wpmg.query_memories(query, limit=max_candidates)
        
        return memories
    
    def fuse_results(self, all_results: Dict[str, List[RetrievalResult]], top_k: int = 10) -> List[FusionResult]:
        """Fuse results from multiple retrieval methods"""
        
        # Collect all unique documents
        doc_scores = defaultdict(lambda: defaultdict(float))
        doc_content = {}
        doc_metadata = defaultdict(dict)
        
        for method, results in all_results.items():
            for result in results:
                doc_scores[result.doc_id][method] = result.score
                doc_content[result.doc_id] = result.content
                doc_metadata[result.doc_id].update(result.metadata)
                doc_metadata[result.doc_id]['retrieved_by'] = method
        
        # Compute fusion scores
        fused_results = []
        for doc_id in doc_scores:
            # Reciprocal Rank Fusion (RRF)
            rrf_score = 0.0
            individual_scores = {}
            
            for method in self.retrievers.keys():
                if method in doc_scores[doc_id]:
                    # Get rank of document in this method's results
                    rank = next(
                        (i + 1 for i, result in enumerate(all_results[method]) 
                         if result.doc_id == doc_id),
                        1000  # Large number if not found
                    )
                    rrf_score += 1.0 / (rank + 60)  # RRF formula with constant k=60
                    individual_scores[method] = doc_scores[doc_id][method]
                else:
                    individual_scores[method] = 0.0
            
            # Weighted fusion score
            weighted_score = sum(
                self.fusion_weights[method] * individual_scores[method]
                for method in self.retrievers.keys()
            )
            
            # Combine RRF and weighted scores
            final_score = 0.7 * rrf_score + 0.3 * weighted_score
            
            fused_results.append(FusionResult(
                doc_id=doc_id,
                content=doc_content[doc_id],
                fusion_score=final_score,
                individual_scores=individual_scores,
                metadata=doc_metadata[doc_id]
            ))
        
        # Sort by fusion score and return top-k
        fused_results.sort(key=lambda x: x.fusion_score, reverse=True)
        return fused_results[:top_k]
    
    def retrieve(self, query: str, top_k: int = 10) -> List[FusionResult]:
        """Main retrieval function combining all methods"""
        
        start_time = time.time()
        
        # Get candidate memories
        candidates = self.get_candidates(query, max_candidates=top_k * 10)
        
        if not candidates:
            self.logger.info("No candidate memories found")
            return []
        
        # Run all retrieval methods
        all_results = {}
        
        for method_name, retriever in self.retrievers.items():
            try:
                results = retriever.retrieve(query, candidates, k=top_k * 2)
                all_results[method_name] = results
                
                self.logger.info(f"{method_name}: {len(results)} results")
                
            except Exception as e:
                self.logger.error(f"{method_name} retrieval failed: {e}")
                all_results[method_name] = []
        
        # Fuse results
        fused_results = self.fuse_results(all_results, top_k=top_k)
        
        processing_time = time.time() - start_time
        
        self.logger.info(
            f"RAG Fusion complete: {len(fused_results)} results in {processing_time:.3f}s"
        )
        
        return fused_results
    
    def explain_retrieval(self, query: str, result: FusionResult) -> Dict[str, Any]:
        """Explain why a particular result was retrieved"""
        
        explanations = {}
        
        # Individual method explanations
        for method, score in result.individual_scores.items():
            if score > 0:
                retriever = self.retrievers[method]
                method_explanation = {
                    'score': score,
                    'weight': self.fusion_weights[method],
                    'contribution': score * self.fusion_weights[method]
                }
                
                # Add method-specific details
                if method == 'dense':
                    method_explanation['similarity_type'] = 'cosine_embedding'
                elif method == 'sparse':
                    method_explanation['similarity_type'] = 'tfidf_cosine'
                elif method == 'graph':
                    method_explanation['centrality'] = result.metadata.get('importance', 0)
                elif method == 'colbert':
                    method_explanation['interaction_type'] = 'late_interaction'
                
                explanations[method] = method_explanation
        
        return {
            'query': query,
            'fusion_score': result.fusion_score,
            'method_explanations': explanations,
            'total_methods_used': len([s for s in result.individual_scores.values() if s > 0])
        }
    
    def update_fusion_weights(self, feedback_data: List[Dict]):
        """Update fusion weights based on user feedback"""
        
        # Simple weight update based on feedback
        method_performance = defaultdict(list)
        
        for feedback in feedback_data:
            query = feedback['query']
            relevant_docs = feedback['relevant_docs']
            
            # Get retrieval results for this query
            results = self.retrieve(query, top_k=20)
            
            # Evaluate each method
            for method in self.retrievers.keys():
                method_hits = sum(1 for result in results 
                                if result.doc_id in relevant_docs 
                                and result.individual_scores[method] > 0)
                
                precision = method_hits / len(results) if results else 0
                method_performance[method].append(precision)
        
        # Update weights based on average performance
        for method in self.fusion_weights:
            if method in method_performance and method_performance[method]:
                avg_performance = np.mean(method_performance[method])
                # Smooth update
                self.fusion_weights[method] = 0.8 * self.fusion_weights[method] + 0.2 * avg_performance
        
        # Normalize weights
        total_weight = sum(self.fusion_weights.values())
        for method in self.fusion_weights:
            self.fusion_weights[method] /= total_weight
        
        self.logger.info(f"Updated fusion weights: {self.fusion_weights}")

# Global RAG fusion instances
_global_fusion_systems = {}

def get_rag_fusion(user_id: str) -> RAGFusion:
    """Get RAG fusion system for user"""
    if user_id not in _global_fusion_systems:
        _global_fusion_systems[user_id] = RAGFusion(user_id)
    return _global_fusion_systems[user_id]

def main():
    """Demo RAG Fusion system"""
    
    logging.basicConfig(level=logging.INFO)
    
    # Setup demo user
    user_id = "demo_user_123"
    
    # Initialize WPMG with some memories
    wpmg = get_wpmg_builder(user_id)
    
    # Add diverse memories
    memories = [
        ("Python is a versatile programming language used for web development, data science, and AI", "fact"),
        ("I learned Python programming last year through online courses", "experience"),
        ("My goal is to become proficient in machine learning and AI", "goal"),
        ("I prefer working on challenging technical problems", "preference"),
        ("Currently exploring different ML frameworks like TensorFlow and PyTorch", "context"),
        ("Deep learning requires significant computational resources", "fact"),
        ("I attended a neural networks workshop at Stanford", "experience"),
        ("Planning to build a personal AI assistant project", "goal"),
    ]
    
    print("🧠 Initializing RAG Fusion System...")
    print("Adding memories to WPMG:\n")
    
    for content, content_type in memories:
        wpmg.add_memory(content, content_type, "demo")
        print(f"  ✅ Added: {content[:60]}...")
    
    # Initialize RAG Fusion
    rag_fusion = get_rag_fusion(user_id)
    
    # Test queries
    test_queries = [
        "What programming languages do I know?",
        "Tell me about my AI learning journey",
        "What are my current technical goals?",
        "How do I approach learning new technologies?"
    ]
    
    print(f"\n🔍 Testing RAG Fusion with sample queries:\n")
    
    for i, query in enumerate(test_queries, 1):
        print(f"Query {i}: {query}")
        
        # Retrieve using RAG Fusion
        results = rag_fusion.retrieve(query, top_k=3)
        
        print(f"  Found {len(results)} relevant memories:")
        for j, result in enumerate(results, 1):
            print(f"    {j}. {result.content[:80]}...")
            print(f"       Fusion Score: {result.fusion_score:.3f}")
            print(f"       Methods: {', '.join(m for m, s in result.individual_scores.items() if s > 0)}")
        
        print()
    
    # Show fusion weights
    print(f"📊 Current Fusion Weights:")
    for method, weight in rag_fusion.fusion_weights.items():
        print(f"  {method}: {weight:.3f}")
    
    print(f"\n✅ RAG Fusion demonstration complete!")

if __name__ == "__main__":
    main()