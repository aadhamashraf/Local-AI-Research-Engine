"""
Vector Store - ChromaDB wrapper for semantic search
"""

import chromadb
from chromadb.config import Settings
from typing import List, Dict, Any, Optional
from loguru import logger
from pathlib import Path
import uuid


class VectorStore:
    """Manage vector embeddings using ChromaDB."""
    
    def __init__(
        self,
        persist_directory: str = "./data/embeddings",
        collection_name: str = "research_documents",
        distance_metric: str = "cosine"
    ):
        """
        Initialize vector store.
        
        Args:
            persist_directory: Directory for persistent storage
            collection_name: Name of the collection
            distance_metric: Distance metric (cosine, l2, ip)
        """
        self.persist_directory = Path(persist_directory)
        self.persist_directory.mkdir(parents=True, exist_ok=True)
        
        self.collection_name = collection_name
        self.distance_metric = distance_metric
        
        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(
            path=str(self.persist_directory)
        )
        
        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": distance_metric}
        )
        
        logger.info(f"Initialized vector store: {collection_name}")
        logger.info(f"Collection size: {self.collection.count()} documents")
    
    def add_documents(
        self,
        chunks: List[Dict[str, Any]],
        embeddings: List[List[float]]
    ) -> List[str]:
        """
        Add documents with embeddings to the store.
        
        Args:
            chunks: List of chunk dictionaries
            embeddings: List of embedding vectors
            
        Returns:
            List of document IDs
        """
        if len(chunks) != len(embeddings):
            raise ValueError("Number of chunks must match number of embeddings")
        
        if not chunks:
            return []
        
        # Generate IDs
        ids = [str(uuid.uuid4()) for _ in chunks]
        
        # Extract content and metadata
        documents = [chunk["content"] for chunk in chunks]
        metadatas = []
        
        for chunk in chunks:
            # Flatten metadata for ChromaDB (only supports flat dicts)
            metadata = chunk.get("metadata", {})
            flat_metadata = self._flatten_metadata(metadata)
            metadatas.append(flat_metadata)
        
        # Add to collection
        self.collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas
        )
        
        logger.info(f"Added {len(chunks)} documents to vector store")
        return ids
    
    def search(
        self,
        query_embedding: List[float],
        top_k: int = 10,
        filter_dict: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Search for similar documents.
        
        Args:
            query_embedding: Query embedding vector
            top_k: Number of results to return
            filter_dict: Optional metadata filters
            
        Returns:
            List of results with content, metadata, and scores
        """
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=filter_dict
        )
        
        # Format results
        formatted_results = []
        
        for i in range(len(results['ids'][0])):
            formatted_results.append({
                "id": results['ids'][0][i],
                "content": results['documents'][0][i],
                "metadata": results['metadatas'][0][i],
                "score": 1.0 - results['distances'][0][i],  # Convert distance to similarity
                "distance": results['distances'][0][i]
            })
        
        logger.debug(f"Vector search returned {len(formatted_results)} results")
        return formatted_results
    
    def delete_by_source(self, source: str) -> int:
        """
        Delete all documents from a specific source.
        
        Args:
            source: Source file name
            
        Returns:
            Number of documents deleted
        """
        # Get all documents from this source
        results = self.collection.get(
            where={"source": source}
        )
        
        if results['ids']:
            self.collection.delete(ids=results['ids'])
            logger.info(f"Deleted {len(results['ids'])} documents from {source}")
            return len(results['ids'])
        
        return 0
    
    def get_all_sources(self) -> List[str]:
        """Get list of all unique sources in the collection."""
        all_docs = self.collection.get()
        
        sources = set()
        for metadata in all_docs.get('metadatas', []):
            if 'source' in metadata:
                sources.add(metadata['source'])
        
        return sorted(list(sources))
    
    def clear(self):
        """Clear all documents from the collection."""
        self.client.delete_collection(self.collection_name)
        self.collection = self.client.create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": self.distance_metric}
        )
        logger.info("Cleared vector store")
    
    def count(self) -> int:
        """Get total number of documents."""
        return self.collection.count()
    
    def _flatten_metadata(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Flatten nested metadata for ChromaDB.
        
        ChromaDB only supports: str, int, float, bool
        """
        flat = {}
        
        for key, value in metadata.items():
            if isinstance(value, (str, int, float, bool)):
                flat[key] = value
            elif isinstance(value, dict):
                # Skip nested dicts (like entities)
                continue
            elif isinstance(value, list):
                # Convert lists to comma-separated strings
                if value and isinstance(value[0], str):
                    flat[key] = ",".join(value)
            else:
                # Convert to string
                flat[key] = str(value)
        
        return flat
