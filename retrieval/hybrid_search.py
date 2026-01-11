"""
Hybrid Search - Combines vector, keyword, and graph search
"""

from typing import List, Dict, Any, Optional
from loguru import logger

from index.vector_store import VectorStore
from index.keyword_index import KeywordIndex
from index.knowledge_graph import KnowledgeGraph
from llm.ollama_client import OllamaClient


class HybridRetriever:
    """Hybrid retrieval combining multiple search strategies."""
    
    def __init__(
        self,
        vector_store: VectorStore,
        keyword_index: KeywordIndex,
        knowledge_graph: KnowledgeGraph,
        ollama_client: OllamaClient,
        vector_weight: float = 0.5,
        keyword_weight: float = 0.3,
        graph_weight: float = 0.2
    ):
        """
        Initialize hybrid retriever.
        
        Args:
            vector_store: Vector store instance
            keyword_index: Keyword index instance
            knowledge_graph: Knowledge graph instance
            ollama_client: Ollama client for embeddings
            vector_weight: Weight for vector search scores
            keyword_weight: Weight for keyword search scores
            graph_weight: Weight for graph search scores
        """
        self.vector_store = vector_store
        self.keyword_index = keyword_index
        self.knowledge_graph = knowledge_graph
        self.client = ollama_client
        
        self.vector_weight = vector_weight
        self.keyword_weight = keyword_weight
        self.graph_weight = graph_weight
        
        logger.info("Initialized hybrid retriever")
    
    def retrieve(
        self,
        query: str,
        vector_top_k: int = 20,
        keyword_top_k: int = 20,
        graph_expansion: bool = True,
        final_top_k: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Retrieve relevant documents using hybrid search.
        
        Args:
            query: Search query
            vector_top_k: Number of results from vector search
            keyword_top_k: Number of results from keyword search
            graph_expansion: Whether to expand with graph
            final_top_k: Final number of results to return
            
        Returns:
            List of retrieved documents with scores
        """
        logger.info(f"Hybrid retrieval for query: {query}")
        
        # 1. Vector search
        vector_results = self._vector_search(query, vector_top_k)
        logger.debug(f"Vector search: {len(vector_results)} results")
        
        # 2. Keyword search
        keyword_results = self._keyword_search(query, keyword_top_k)
        logger.debug(f"Keyword search: {len(keyword_results)} results")
        
        # 3. Graph expansion (optional)
        graph_results = []
        if graph_expansion:
            graph_results = self._graph_expansion(query, vector_results)
            logger.debug(f"Graph expansion: {len(graph_results)} results")
        
        # 4. Merge and deduplicate
        merged = self._merge_results(
            vector_results,
            keyword_results,
            graph_results
        )
        
        # 5. Reciprocal Rank Fusion (RRF)
        fused = self._reciprocal_rank_fusion(merged)
        
        # 6. Return top-k
        final_results = fused[:final_top_k]
        
        logger.info(f"Hybrid retrieval returned {len(final_results)} results")
        return final_results
    
    def _vector_search(
        self,
        query: str,
        top_k: int
    ) -> List[Dict[str, Any]]:
        """Perform vector similarity search."""
        
        # Generate query embedding
        query_embedding = self.client.embed(query)
        
        # Search
        results = self.vector_store.search(
            query_embedding=query_embedding,
            top_k=top_k
        )
        
        # Add source tag
        for result in results:
            result["search_source"] = "vector"
        
        return results
    
    def _keyword_search(
        self,
        query: str,
        top_k: int
    ) -> List[Dict[str, Any]]:
        """Perform keyword search."""
        
        results = self.keyword_index.search(
            query=query,
            top_k=top_k
        )
        
        # Add source tag
        for result in results:
            result["search_source"] = "keyword"
        
        return results
    
    def _graph_expansion(
        self,
        query: str,
        initial_results: List[Dict[str, Any]],
        max_expansion: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Expand results using knowledge graph.
        
        Find related entities from top results and retrieve their documents.
        """
        expanded = []
        
        # Extract entities from top results
        seen_entities = set()
        
        for result in initial_results[:5]:  # Use top 5 for expansion
            entities = result.get("metadata", {}).get("entities", {})
            
            for entity_type, entity_list in entities.items():
                if isinstance(entity_list, str):
                    entity_list = entity_list.split(",")
                
                for entity_name in entity_list:
                    entity_name = entity_name.strip()
                    if entity_name and entity_name not in seen_entities:
                        seen_entities.add(entity_name)
                        
                        # Search for this entity in the graph
                        entity_matches = self.knowledge_graph.search_entities(
                            query=entity_name,
                            limit=1
                        )
                        
                        if entity_matches:
                            entity_id = entity_matches[0]["entity_id"]
                            
                            # Get related entities
                            related = self.knowledge_graph.get_related_entities(
                                entity_id=entity_id,
                                max_depth=1,
                                max_results=5
                            )
                            
                            for rel in related:
                                expanded.append({
                                    "entity": rel["label"],
                                    "relationship": rel["relationship"],
                                    "search_source": "graph"
                                })
        
        return expanded[:max_expansion]
    
    def _merge_results(
        self,
        vector_results: List[Dict[str, Any]],
        keyword_results: List[Dict[str, Any]],
        graph_results: List[Dict[str, Any]]
    ) -> Dict[str, Dict[str, Any]]:
        """
        Merge results from different sources.
        
        Returns:
            Dictionary mapping document IDs to result data
        """
        merged = {}
        
        # Add vector results
        for i, result in enumerate(vector_results):
            doc_id = result.get("id")
            if doc_id:
                merged[doc_id] = result.copy()
                merged[doc_id]["vector_rank"] = i
                merged[doc_id]["vector_score"] = result.get("score", 0)
        
        # Add keyword results
        for i, result in enumerate(keyword_results):
            doc_id = result.get("id")
            if doc_id:
                if doc_id in merged:
                    merged[doc_id]["keyword_rank"] = i
                    merged[doc_id]["keyword_score"] = result.get("score", 0)
                else:
                    merged[doc_id] = result.copy()
                    merged[doc_id]["keyword_rank"] = i
                    merged[doc_id]["keyword_score"] = result.get("score", 0)
        
        # Note: Graph results are used for context but don't directly contribute documents
        # They influence the final ranking through entity matching
        
        return merged
    
    def _reciprocal_rank_fusion(
        self,
        merged: Dict[str, Dict[str, Any]],
        k: int = 60
    ) -> List[Dict[str, Any]]:
        """
        Apply Reciprocal Rank Fusion to merge rankings.
        
        RRF score = sum(1 / (k + rank)) for each ranking
        
        Args:
            merged: Merged results dictionary
            k: RRF constant (typically 60)
            
        Returns:
            Sorted list of results
        """
        for doc_id, result in merged.items():
            rrf_score = 0.0
            
            # Vector ranking
            if "vector_rank" in result:
                rrf_score += self.vector_weight / (k + result["vector_rank"])
            
            # Keyword ranking
            if "keyword_rank" in result:
                rrf_score += self.keyword_weight / (k + result["keyword_rank"])
            
            result["rrf_score"] = rrf_score
        
        # Sort by RRF score
        sorted_results = sorted(
            merged.values(),
            key=lambda x: x.get("rrf_score", 0),
            reverse=True
        )
        
        return sorted_results
    
    def get_retrieval_stats(
        self,
        results: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Get statistics about retrieval results."""
        
        sources = {}
        for result in results:
            source = result.get("search_source", "unknown")
            sources[source] = sources.get(source, 0) + 1
        
        return {
            "total_results": len(results),
            "sources": sources,
            "avg_rrf_score": sum(r.get("rrf_score", 0) for r in results) / len(results) if results else 0
        }
