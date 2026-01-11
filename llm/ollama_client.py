"""
Ollama Client - Wrapper for Ollama API
Provides text generation and embedding capabilities.
"""

import ollama
import asyncio
from typing import List, Dict, Any, Optional, Generator
from loguru import logger
import time
import concurrent.futures


class OllamaClient:
    """Client for interacting with Ollama models (Sync and Async)."""
    
    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        llm_model: str = "mistral:latest",
        embedding_model: str = "nomic-embed-text",
        timeout: int = 120,
        max_retries: int = 3
    ):
        """
        Initialize Ollama client.
        
        Args:
            base_url: Ollama server URL
            llm_model: Model name for text generation
            embedding_model: Model name for embeddings
            timeout: Request timeout in seconds
            max_retries: Maximum number of retry attempts
        """
        self.base_url = base_url
        self.llm_model = llm_model
        self.embedding_model = embedding_model
        self.timeout = timeout
        self.max_retries = max_retries
        
        # Configure ollama client
        self.client = ollama.Client(host=base_url)
        
        logger.info(f"Initialized Ollama client: {base_url}")
        logger.info(f"LLM Model: {llm_model}, Embedding Model: {embedding_model}")
    
    def generate(
        self,
        prompt: str,
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 2000,
        system_prompt: Optional[str] = None,
        stop: Optional[List[str]] = None
    ) -> str:
        """
        Generate text using the LLM.
        
        Args:
            prompt: User prompt
            model: Model to use (defaults to llm_model)
            temperature: Sampling temperature (0.0 to 1.0)
            max_tokens: Maximum tokens to generate
            system_prompt: Optional system prompt
            stop: Optional stop sequences
            
        Returns:
            Generated text
        """
        model = model or self.llm_model
        
        for attempt in range(self.max_retries):
            try:
                messages = []
                
                if system_prompt:
                    messages.append({
                        "role": "system",
                        "content": system_prompt
                    })
                
                messages.append({
                    "role": "user",
                    "content": prompt
                })
                
                logger.debug(f"Generating with {model} (attempt {attempt + 1}/{self.max_retries})")
                
                response = self.client.chat(
                    model=model,
                    messages=messages,
                    options={
                        "temperature": temperature,
                        "num_predict": max_tokens,
                        "stop": stop or []
                    }
                )
                
                result = response['message']['content']
                logger.debug(f"Generated {len(result)} characters")
                return result
                
            except Exception as e:
                logger.warning(f"Generation attempt {attempt + 1} failed: {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
                else:
                    logger.error(f"All generation attempts failed for model {model}")
                    raise
    
    def stream_generate(
        self,
        prompt: str,
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 2000,
        system_prompt: Optional[str] = None
    ) -> Generator[str, None, None]:
        """
        Stream text generation.
        
        Args:
            prompt: User prompt
            model: Model to use
            temperature: Sampling temperature
            max_tokens: Maximum tokens
            system_prompt: Optional system prompt
            
        Yields:
            Text chunks as they are generated
        """
        model = model or self.llm_model
        
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        try:
            stream = self.client.chat(
                model=model,
                messages=messages,
                stream=True,
                options={
                    "temperature": temperature,
                    "num_predict": max_tokens
                }
            )
            
            for chunk in stream:
                if 'message' in chunk and 'content' in chunk['message']:
                    yield chunk['message']['content']
                    
        except Exception as e:
            logger.error(f"Streaming generation failed: {e}")
            raise
    
    def embed(
        self,
        text: str,
        model: Optional[str] = None
    ) -> List[float]:
        """
        Generate embeddings for text.
        
        Args:
            text: Text to embed
            model: Embedding model to use
            
        Returns:
            Embedding vector
        """
        model = model or self.embedding_model
        
        for attempt in range(self.max_retries):
            try:
                response = self.client.embeddings(
                    model=model,
                    prompt=text
                )
                
                embedding = response['embedding']
                logger.debug(f"Generated embedding of dimension {len(embedding)}")
                return embedding
                
            except Exception as e:
                logger.warning(f"Embedding attempt {attempt + 1} failed: {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(2 ** attempt)
                else:
                    logger.error(f"All embedding attempts failed")
                    raise
    
    def embed_batch(
        self,
        texts: List[str],
        model: Optional[str] = None,
        batch_size: int = 10,
        max_workers: int = 4
    ) -> List[List[float]]:
        """
        Generate embeddings for multiple texts in parallel.
        
        Args:
            texts: List of texts to embed
            model: Embedding model to use
            batch_size: Number of texts to process at once (not used in parallel mode efficiently but kept for compat)
            max_workers: Number of parallel threads
            
        Returns:
            List of embedding vectors in correct order
        """
        import concurrent.futures
        
        embeddings = [None] * len(texts)
        model = model or self.embedding_model
        
        logger.info(f"Generating embeddings for {len(texts)} texts with {max_workers} workers")
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Create a future for each text
            # Note: We process individual texts in parallel rather than batches
            # because Ollama handles concurrency well
            future_to_idx = {
                executor.submit(self.embed, text, model): i 
                for i, text in enumerate(texts)
            }
            
            for future in concurrent.futures.as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    embedding = future.result()
                    embeddings[idx] = embedding
                except Exception as e:
                    logger.error(f"Embedding failed for text index {idx}: {e}")
                    # Placeholder zero vector or re-raise? 
                    # For robustness, we'll try to re-run locally or raise
                    raise
        
        return embeddings
    
    async def agenerate(
        self,
        prompt: str,
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 2000,
        system_prompt: Optional[str] = None
    ) -> str:
        """
        Async generate text using the LLM.
        """
        model = model or self.llm_model
        client = ollama.AsyncClient(host=self.base_url)
        
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        try:
            response = await client.chat(
                model=model,
                messages=messages,
                options={
                    "temperature": temperature,
                    "num_predict": max_tokens
                }
            )
            return response['message']['content']
        except Exception as e:
            logger.error(f"Async generation failed: {e}")
            raise

    async def aembed_batch(
        self,
        texts: List[str],
        model: Optional[str] = None
    ) -> List[List[float]]:
        """
        Async generate embeddings for multiple texts (High Concurrency).
        """
        model = model or self.embedding_model
        client = ollama.AsyncClient(host=self.base_url)
        
        try:
            tasks = []
            for text in texts:
                tasks.append(client.embeddings(model=model, prompt=text))
            
            # Run all embedding requests concurrently
            logger.info(f"Async embedding {len(texts)} texts...")
            responses = await asyncio.gather(*tasks)
            
            return [r['embedding'] for r in responses]
            
        except Exception as e:
            logger.error(f"Async batch embedding failed: {e}")
            raise
    
    def check_model_availability(self, model: str) -> bool:
        """
        Check if a model is available.
        
        Args:
            model: Model name to check
            
        Returns:
            True if model is available
        """
        try:
            models_response = self.client.list()
            
            # Extract models list regardless of response format (Object vs Dict)
            if hasattr(models_response, 'models'):
                models_list = models_response.models
            else:
                models_list = models_response.get('models', [])
            
            available_models = []
            for m in models_list:
                # Try multiple ways to get the model name
                name = None
                if hasattr(m, 'model'):
                    name = m.model
                elif hasattr(m, 'name'):
                    name = m.name
                elif isinstance(m, dict):
                    name = m.get('model') or m.get('name')
                
                if name:
                    available_models.append(name)
            
            # Check for exact match or partial match (e.g., "qwen2.5:7b" in "qwen2.5:7b-instruct")
            is_available = any(model in m for m in available_models)
            
            if is_available:
                logger.info(f"Model {model} is available")
            else:
                logger.warning(f"Model {model} not found. Available: {available_models}")
            
            return is_available
            
        except Exception as e:
            logger.error(f"Failed to check model availability: {e}")
            return False
    
    def verify_setup(self) -> Dict[str, bool]:
        """
        Verify that Ollama is running and models are available.
        
        Returns:
            Dictionary with verification results
        """
        results = {
            "ollama_running": False,
            "llm_available": False,
            "embedding_available": False
        }
        
        try:
            # Check if Ollama is running
            self.client.list()
            results["ollama_running"] = True
            logger.info("✓ Ollama is running")
            
            # Check LLM model
            results["llm_available"] = self.check_model_availability(self.llm_model)
            if results["llm_available"]:
                logger.info(f"✓ LLM model '{self.llm_model}' is available")
            else:
                logger.error(f"✗ LLM model '{self.llm_model}' not found")
            
            # Check embedding model
            results["embedding_available"] = self.check_model_availability(self.embedding_model)
            if results["embedding_available"]:
                logger.info(f"✓ Embedding model '{self.embedding_model}' is available")
            else:
                logger.error(f"✗ Embedding model '{self.embedding_model}' not found")
                
        except Exception as e:
            logger.error(f"✗ Ollama is not running or not accessible: {e}")
        
        return results
