"""
Keyword Index - BM25-based keyword search
"""

from rank_bm25 import BM25Okapi
from typing import List, Dict, Any
from loguru import logger
import pickle
from pathlib import Path
import re


class KeywordIndex:
    """BM25-based keyword search index."""
    
    def __init__(
        self,
        persist_path: str = "./data/keyword_index.pkl",
        k1: float = 1.5,
        b: float = 0.75
    ):
        """
        Initialize keyword index.
        
        Args:
            persist_path: Path to save/load index
            k1: BM25 term frequency saturation parameter
            b: BM25 length normalization parameter
        """
        self.persist_path = Path(persist_path)
        self.k1 = k1
        self.b = b
        
        self.bm25 = None
        self.documents = []  # Store original documents
        self.doc_ids = []    # Store document IDs
        self.tokenized_corpus = []
        
        logger.info("Initialized keyword index")
    
    def build_index(
        self,
        chunks: List[Dict[str, Any]],
        doc_ids: List[str]
    ):
        """
        Build BM25 index from chunks.
        
        Args:
            chunks: List of chunk dictionaries
            doc_ids: List of document IDs (must match chunks)
        """
        if len(chunks) != len(doc_ids):
            raise ValueError("Number of chunks must match number of IDs")
        
        self.documents = chunks
        self.doc_ids = doc_ids
        
        # Tokenize documents
        self.tokenized_corpus = [
            self._tokenize(chunk["content"])
            for chunk in chunks
        ]
        
        # Build BM25 index
        self.bm25 = BM25Okapi(
            self.tokenized_corpus,
            k1=self.k1,
            b=self.b
        )
        
        logger.info(f"Built BM25 index with {len(chunks)} documents")
    
    def search(
        self,
        query: str,
        top_k: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Search for documents matching the query.
        
        Args:
            query: Search query
            top_k: Number of results to return
            
        Returns:
            List of results with content, metadata, and scores
        """
        if self.bm25 is None:
            logger.warning("Index not built, returning empty results")
            return []
        
        # Tokenize query
        tokenized_query = self._tokenize(query)
        
        # Get scores
        scores = self.bm25.get_scores(tokenized_query)
        
        # Get top-k indices
        top_indices = sorted(
            range(len(scores)),
            key=lambda i: scores[i],
            reverse=True
        )[:top_k]
        
        # Format results
        results = []
        for idx in top_indices:
            if scores[idx] > 0:  # Only include non-zero scores
                results.append({
                    "id": self.doc_ids[idx],
                    "content": self.documents[idx]["content"],
                    "metadata": self.documents[idx]["metadata"],
                    "score": float(scores[idx])
                })
        
        logger.debug(f"Keyword search returned {len(results)} results")
        return results
    
    def add_documents(
        self,
        chunks: List[Dict[str, Any]],
        doc_ids: List[str]
    ):
        """
        Add new documents to existing index.
        
        Args:
            chunks: New chunks to add
            doc_ids: Document IDs for new chunks
        """
        if not chunks:
            return
        
        # Add to existing documents
        self.documents.extend(chunks)
        self.doc_ids.extend(doc_ids)
        
        # Tokenize new documents
        new_tokenized = [self._tokenize(chunk["content"]) for chunk in chunks]
        self.tokenized_corpus.extend(new_tokenized)
        
        # Rebuild index (BM25 doesn't support incremental updates)
        self.bm25 = BM25Okapi(
            self.tokenized_corpus,
            k1=self.k1,
            b=self.b
        )
        
        logger.info(f"Added {len(chunks)} documents to keyword index")
    
    def save(self):
        """Save index to disk."""
        self.persist_path.parent.mkdir(parents=True, exist_ok=True)
        
        data = {
            "documents": self.documents,
            "doc_ids": self.doc_ids,
            "tokenized_corpus": self.tokenized_corpus,
            "k1": self.k1,
            "b": self.b
        }
        
        with open(self.persist_path, 'wb') as f:
            pickle.dump(data, f)
        
        logger.info(f"Saved keyword index to {self.persist_path}")
    
    def load(self) -> bool:
        """
        Load index from disk.
        
        Returns:
            True if loaded successfully, False otherwise
        """
        if not self.persist_path.exists():
            logger.warning(f"Index file not found: {self.persist_path}")
            return False
        
        try:
            with open(self.persist_path, 'rb') as f:
                data = pickle.load(f)
            
            self.documents = data["documents"]
            self.doc_ids = data["doc_ids"]
            self.tokenized_corpus = data["tokenized_corpus"]
            self.k1 = data.get("k1", 1.5)
            self.b = data.get("b", 0.75)
            
            # Rebuild BM25 object
            self.bm25 = BM25Okapi(
                self.tokenized_corpus,
                k1=self.k1,
                b=self.b
            )
            
            logger.info(f"Loaded keyword index with {len(self.documents)} documents")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load index: {e}")
            return False
    
    def clear(self):
        """Clear the index."""
        self.bm25 = None
        self.documents = []
        self.doc_ids = []
        self.tokenized_corpus = []
        logger.info("Cleared keyword index")
    
    def count(self) -> int:
        """Get number of documents in index."""
        return len(self.documents)
    
    def _tokenize(self, text: str) -> List[str]:
        """
        Tokenize text for BM25.
        
        Simple tokenization:
        - Lowercase
        - Split on non-alphanumeric
        - Remove short tokens
        """
        # Lowercase
        text = text.lower()
        
        # Split on non-alphanumeric (but keep hyphens in words)
        tokens = re.findall(r'\b[\w-]+\b', text)
        
        # Filter short tokens
        tokens = [t for t in tokens if len(t) > 2]
        
        return tokens
