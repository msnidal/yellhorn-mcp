"""Unified LLM Manager with automatic chunking support."""

import asyncio
import json
import re
from typing import Any, Dict, List, Optional, Union
from openai import AsyncOpenAI
from google import genai
from .token_counter import TokenCounter


class UsageMetadata:
    """
    Unified usage metadata class that handles both OpenAI and Gemini formats.
    
    This class provides a consistent interface for accessing token usage information
    regardless of the source (OpenAI API, Gemini API, or dictionary).
    """
    
    def __init__(self, data: Any = None):
        """
        Initialize UsageMetadata from various sources.
        
        Args:
            data: Can be:
                - OpenAI CompletionUsage object
                - Gemini GenerateContentResponseUsageMetadata object
                - Dictionary with token counts
                - None (defaults to 0 for all values)
        """
        self.prompt_tokens: int = 0
        self.completion_tokens: int = 0
        self.total_tokens: int = 0
        self.model: Optional[str] = None
        
        if data is None:
            return
            
        if isinstance(data, dict):
            # Handle dictionary format (our internal format)
            self.prompt_tokens = data.get("prompt_tokens", 0)
            self.completion_tokens = data.get("completion_tokens", 0)
            self.total_tokens = data.get("total_tokens", 0)
            self.model = data.get("model")
        elif hasattr(data, "prompt_tokens"):
            # OpenAI CompletionUsage format
            self.prompt_tokens = getattr(data, "prompt_tokens", 0)
            self.completion_tokens = getattr(data, "completion_tokens", 0)
            self.total_tokens = getattr(data, "total_tokens", 0)
        elif hasattr(data, "prompt_token_count"):
            # Gemini GenerateContentResponseUsageMetadata format
            self.prompt_tokens = getattr(data, "prompt_token_count", 0)
            self.completion_tokens = getattr(data, "candidates_token_count", 0)
            self.total_tokens = getattr(data, "total_token_count", 0)
    
    @property
    def prompt_token_count(self) -> int:
        """Gemini-style property for compatibility."""
        return self.prompt_tokens
    
    @property
    def candidates_token_count(self) -> int:
        """Gemini-style property for compatibility."""
        return self.completion_tokens
    
    @property
    def total_token_count(self) -> int:
        """Gemini-style property for compatibility."""
        return self.total_tokens
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        result = {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
        }
        if self.model:
            result["model"] = self.model
        return result
    
    def __bool__(self) -> bool:
        """Check if we have valid usage data."""
        return self.total_tokens > 0


class ChunkingStrategy:
    """Strategies for splitting text into chunks."""
    
    @staticmethod
    def split_by_sentences(
        text: str, 
        max_tokens: int, 
        token_counter: TokenCounter,
        model: str,
        overlap_ratio: float = 0.1
    ) -> List[str]:
        """
        Split text into chunks by sentence boundaries.
        
        Args:
            text: Text to split
            max_tokens: Maximum tokens per chunk
            token_counter: TokenCounter instance
            model: Model name for token counting
            overlap_ratio: Ratio of overlap between chunks (0.0 to 0.5)
            
        Returns:
            List of text chunks
        """
        # Split by sentence-ending punctuation
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        chunks = []
        current_chunk = []
        current_tokens = 0
        overlap_tokens = int(max_tokens * overlap_ratio)
        
        for sentence in sentences:
            sentence_tokens = token_counter.count_tokens(sentence, model)
            
            # If single sentence exceeds max tokens, split it further
            if sentence_tokens > max_tokens:
                # Split by clauses or words
                words = sentence.split()
                word_chunk = []
                word_tokens = 0
                
                for word in words:
                    word_token_count = token_counter.count_tokens(word + " ", model)
                    if word_tokens + word_token_count > max_tokens and word_chunk:
                        chunks.append(" ".join(word_chunk))
                        # Add overlap from the end of previous chunk
                        overlap_words = []
                        overlap_token_count = 0
                        for w in reversed(word_chunk):
                            w_tokens = token_counter.count_tokens(w + " ", model)
                            if overlap_token_count + w_tokens <= overlap_tokens:
                                overlap_words.insert(0, w)
                                overlap_token_count += w_tokens
                            else:
                                break
                        word_chunk = overlap_words + [word]
                        word_tokens = overlap_token_count + word_token_count
                    else:
                        word_chunk.append(word)
                        word_tokens += word_token_count
                
                if word_chunk:
                    chunks.append(" ".join(word_chunk))
            
            # Normal sentence processing
            elif current_tokens + sentence_tokens > max_tokens and current_chunk:
                # Save current chunk
                chunks.append(" ".join(current_chunk))
                
                # Calculate overlap sentences
                overlap_sentences = []
                overlap_token_count = 0
                for sent in reversed(current_chunk):
                    sent_tokens = token_counter.count_tokens(sent, model)
                    if overlap_token_count + sent_tokens <= overlap_tokens:
                        overlap_sentences.insert(0, sent)
                        overlap_token_count += sent_tokens
                    else:
                        break
                
                # Start new chunk with overlap
                current_chunk = overlap_sentences + [sentence]
                current_tokens = overlap_token_count + sentence_tokens
            else:
                current_chunk.append(sentence)
                current_tokens += sentence_tokens
        
        # Add final chunk
        if current_chunk:
            chunks.append(" ".join(current_chunk))
        
        return chunks
    
    @staticmethod
    def split_by_paragraphs(
        text: str,
        max_tokens: int,
        token_counter: TokenCounter,
        model: str,
        overlap_ratio: float = 0.1
    ) -> List[str]:
        """Split text by paragraphs with overlap."""
        paragraphs = text.split('\n\n')
        
        chunks = []
        current_chunk = []
        current_tokens = 0
        overlap_tokens = int(max_tokens * overlap_ratio)
        
        for para in paragraphs:
            para_tokens = token_counter.count_tokens(para, model)
            
            if para_tokens > max_tokens:
                # If paragraph is too large, use sentence splitting
                para_chunks = ChunkingStrategy.split_by_sentences(
                    para, max_tokens, token_counter, model, overlap_ratio
                )
                for chunk in para_chunks:
                    chunks.append(chunk)
            elif current_tokens + para_tokens > max_tokens and current_chunk:
                chunks.append("\n\n".join(current_chunk))
                
                # Add overlap
                overlap_paras = []
                overlap_token_count = 0
                for p in reversed(current_chunk):
                    p_tokens = token_counter.count_tokens(p, model)
                    if overlap_token_count + p_tokens <= overlap_tokens:
                        overlap_paras.insert(0, p)
                        overlap_token_count += p_tokens
                    else:
                        break
                
                current_chunk = overlap_paras + [para]
                current_tokens = overlap_token_count + para_tokens
            else:
                current_chunk.append(para)
                current_tokens += para_tokens
        
        if current_chunk:
            chunks.append("\n\n".join(current_chunk))
        
        return chunks


class LLMManager:
    """Unified manager for LLM calls with automatic chunking."""
    
    def __init__(
        self,
        openai_client: Optional[AsyncOpenAI] = None,
        gemini_client: Optional[genai.Client] = None,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize LLM Manager.
        
        Args:
            openai_client: OpenAI client instance
            gemini_client: Gemini client instance  
            config: Configuration dictionary
        """
        self.token_counter = TokenCounter()
        self.openai_client = openai_client
        self.gemini_client = gemini_client
        self.config = config or {}
        
        # Default configuration
        self.safety_margin = self.config.get("safety_margin_tokens", 1000)
        self.overlap_ratio = self.config.get("overlap_ratio", 0.1)
        self.aggregation_strategy = self.config.get("aggregation_strategy", "concatenate")
        self.chunk_strategy = self.config.get("chunk_strategy", "sentences")
        
        # Track usage metadata from last call
        self._last_usage_metadata = None

    def _is_openai_model(self, model: str) -> bool:
        """Check if model is an OpenAI model."""
        openai_prefixes = ["gpt-", "o3", "o4-"]
        return any(model.startswith(prefix) for prefix in openai_prefixes)
    
    def _is_gemini_model(self, model: str) -> bool:
        """Check if model is a Gemini model."""
        return model.startswith("gemini-")
    
    async def call_llm(
        self,
        prompt: str,
        model: str,
        temperature: float = 0.7,
        system_message: Optional[str] = None,
        response_format: Optional[str] = None,
        **kwargs
    ) -> Union[str, Dict[str, Any]]:
        """
        Call LLM with automatic chunking if needed.
        
        Args:
            prompt: The prompt to send
            model: Model name
            temperature: Temperature for generation
            system_message: Optional system message
            response_format: Optional response format (e.g., "json")
            **kwargs: Additional model-specific parameters
            
        Returns:
            Generated response (string or dict if JSON format)
        """
        # Check if chunking is needed
        if not self.token_counter.can_fit_in_context(prompt, model, self.safety_margin):
            return await self._chunked_call(
                prompt, model, temperature, system_message, response_format, **kwargs
            )
        
        # Single call
        return await self._single_call(
            prompt, model, temperature, system_message, response_format, **kwargs
        )
    
    async def _single_call(
        self,
        prompt: str,
        model: str,
        temperature: float,
        system_message: Optional[str],
        response_format: Optional[str],
        **kwargs
    ) -> Union[str, Dict[str, Any]]:
        """Make a single LLM call."""
        if self._is_openai_model(model):
            return await self._call_openai(
                prompt, model, temperature, system_message, response_format, **kwargs
            )
        elif self._is_gemini_model(model):
            return await self._call_gemini(
                prompt, model, temperature, system_message, response_format, **kwargs
            )
        else:
            raise ValueError(f"Unknown model type: {model}")
    
    async def _call_openai(
        self,
        prompt: str,
        model: str,
        temperature: float,
        system_message: Optional[str],
        response_format: Optional[str],
        **kwargs
    ) -> Union[str, Dict[str, Any]]:
        """Call OpenAI API."""
        if not self.openai_client:
            raise ValueError("OpenAI client not initialized")
        
        messages = []
        if system_message:
            messages.append({"role": "system", "content": system_message})
        messages.append({"role": "user", "content": prompt})
        
        params = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            **kwargs
        }
        
        if response_format == "json":
            params["response_format"] = {"type": "json_object"}
        
        response = await self.openai_client.chat.completions.create(**params)
        content = response.choices[0].message.content
        
        # Store usage metadata
        if hasattr(response, 'usage'):
            self._last_usage_metadata = UsageMetadata(response.usage)
        
        if response_format == "json":
            try:
                return json.loads(content)
            except json.JSONDecodeError:
                return {"error": "Failed to parse JSON", "content": content}
        
        return content
    
    async def _call_gemini(
        self,
        prompt: str,
        model: str,
        temperature: float,
        system_message: Optional[str],
        response_format: Optional[str],
        **kwargs
    ) -> Union[str, Dict[str, Any]]:
        """Call Gemini API."""
        if not self.gemini_client:
            raise ValueError("Gemini client not configured")
        
        # Combine system message with prompt if provided
        full_prompt = prompt
        if system_message:
            full_prompt = f"{system_message}\n\n{prompt}"
        
        # Import GenerateContentConfig with fallback
        try:
            from google.genai.types import GenerateContentConfig
            config_class = GenerateContentConfig
        except ImportError:
            # Fallback to dict config
            config_class = dict
        
        # Build config
        config_dict = {
            "temperature": temperature,
            "response_mime_type": "application/json" if response_format == "json" else "text/plain",
        }
        
        # Add any additional kwargs
        config_dict.update(kwargs)
        
        # Create config instance
        if config_class == GenerateContentConfig:
            config = config_class(**config_dict)
        else:
            config = config_dict
        
        # Make the API call
        response = await self.gemini_client.aio.models.generate_content(
            model=f"models/{model}",
            contents=full_prompt,
            config=config
        )
        
        # Extract text from response
        if hasattr(response, 'text'):
            content = response.text
        else:
            content = str(response)
        
        # Store usage metadata if available
        if hasattr(response, 'usage_metadata'):
            usage = response.usage_metadata
            self._last_usage_metadata = UsageMetadata(usage)
        
        # Parse JSON if requested
        if response_format == "json":
            # Try to extract JSON from the response
            import re
            json_pattern = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
            json_matches = re.findall(json_pattern, content, re.DOTALL)
            
            if json_matches:
                # Try to parse the largest JSON match
                for json_match in sorted(json_matches, key=len, reverse=True):
                    try:
                        return json.loads(json_match)
                    except:
                        pass
                return {"error": "Failed to parse JSON", "content": content}
        
        # Store the response object for potential citation processing
        if hasattr(response, 'grounding_metadata'):
            # Store metadata in a thread-local or instance variable for retrieval
            self._last_gemini_response = response
        
        return content
    
    async def _chunked_call(
        self,
        prompt: str,
        model: str,
        temperature: float,
        system_message: Optional[str],
        response_format: Optional[str],
        **kwargs
    ) -> Union[str, Dict[str, Any]]:
        """Make chunked LLM calls and aggregate results."""
        # Calculate available tokens for content
        model_limit = self.token_counter.get_model_limit(model)
        system_tokens = self.token_counter.count_tokens(system_message or "", model)
        available_tokens = model_limit - system_tokens - self.safety_margin
        
        # Split prompt into chunks
        chunks = self._chunk_prompt(prompt, model, available_tokens)
        
        # Process chunks
        responses = []
        total_usage = UsageMetadata()
        
        for i, chunk in enumerate(chunks):
            # Add context for multi-chunk processing
            chunk_prompt = chunk
            if len(chunks) > 1:
                chunk_prompt = f"[Chunk {i+1}/{len(chunks)}]\n\n{chunk}"
                if i > 0:
                    chunk_prompt = f"[Continuing from previous chunk...]\n\n{chunk_prompt}"
            
            response = await self._single_call(
                chunk_prompt, model, temperature, system_message, response_format, **kwargs
            )
            responses.append(response)
            
            # Aggregate usage metadata
            if self._last_usage_metadata:
                total_usage.prompt_tokens += self._last_usage_metadata.prompt_tokens
                total_usage.completion_tokens += self._last_usage_metadata.completion_tokens
                total_usage.total_tokens += self._last_usage_metadata.total_tokens
        
        # Store aggregated usage
        self._last_usage_metadata = total_usage
        
        # Aggregate responses
        return self._aggregate_responses(responses, response_format)
    
    def _chunk_prompt(self, prompt: str, model: str, max_tokens: int) -> List[str]:
        """Split prompt into chunks based on strategy."""
        if self.chunk_strategy == "paragraphs":
            return ChunkingStrategy.split_by_paragraphs(
                prompt, max_tokens, self.token_counter, model, self.overlap_ratio
            )
        else:  # default to sentences
            return ChunkingStrategy.split_by_sentences(
                prompt, max_tokens, self.token_counter, model, self.overlap_ratio
            )
    
    def _aggregate_responses(
        self, 
        responses: List[Union[str, Dict]], 
        response_format: Optional[str]
    ) -> Union[str, Dict[str, Any]]:
        """Aggregate multiple responses based on strategy."""
        if response_format == "json":
            # For JSON responses, try to merge
            if all(isinstance(r, dict) for r in responses):
                # Merge dictionaries
                result = {}
                for resp in responses:
                    if isinstance(resp, dict):
                        # Deep merge logic
                        for key, value in resp.items():
                            if key in result:
                                if isinstance(result[key], list) and isinstance(value, list):
                                    result[key].extend(value)
                                elif isinstance(result[key], dict) and isinstance(value, dict):
                                    result[key].update(value)
                                else:
                                    # Create list of values
                                    if not isinstance(result[key], list):
                                        result[key] = [result[key]]
                                    result[key].append(value)
                            else:
                                result[key] = value
                return result
            else:
                # Fallback to list of responses
                return {"chunks": responses}
        
        # For text responses
        if self.aggregation_strategy == "summarize":
            # Would require another LLM call to summarize
            # For now, fall back to concatenation
            pass
        
        # Default: concatenate
        text_responses = [str(r) for r in responses]
        return "\n\n---\n\n".join(text_responses)

    async def call_llm_with_citations(
        self,
        prompt: str,
        model: str,
        temperature: float = 0.7,
        system_message: Optional[str] = None,
        response_format: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Call LLM and return both response and citation metadata if available.
        
        This is specifically useful for Gemini models with search grounding enabled.
        
        Args:
            prompt: The prompt to send
            model: Model name
            temperature: Temperature for generation
            system_message: Optional system message
            response_format: Optional response format (e.g., "json")
            **kwargs: Additional arguments passed to the LLM
            
        Returns:
            Dictionary with 'content', 'usage_metadata', and optionally 'grounding_metadata'
        """
        # Reset last response
        self._last_gemini_response = None
        self._last_usage_metadata = None
        
        # Make the regular call
        content = await self.call_llm(
            prompt=prompt,
            model=model,
            temperature=temperature,
            system_message=system_message,
            response_format=response_format,
            **kwargs
        )
        
        # Build result with content and usage
        result = {
            "content": content,
            "usage_metadata": self._last_usage_metadata if self._last_usage_metadata else UsageMetadata()
        }
        
        # Check if we have grounding metadata from Gemini
        if self._is_gemini_model(model) and hasattr(self, '_last_gemini_response'):
            response = getattr(self, '_last_gemini_response', None)
            if response and hasattr(response, 'grounding_metadata'):
                result["grounding_metadata"] = response.grounding_metadata
                
        return result
    
    async def call_llm_with_usage(
        self,
        prompt: str,
        model: str,
        temperature: float = 0.7,
        system_message: Optional[str] = None,
        response_format: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Call LLM and return both response content and usage metadata.
        
        Args:
            prompt: The prompt to send
            model: Model name
            temperature: Temperature for generation
            system_message: Optional system message
            response_format: Optional response format (e.g., "json")
            **kwargs: Additional arguments passed to the LLM
            
        Returns:
            Dictionary with 'content' and 'usage_metadata' (as UsageMetadata object)
        """
        # Reset usage metadata
        self._last_usage_metadata = None
        
        # Make the regular call
        content = await self.call_llm(
            prompt=prompt,
            model=model,
            temperature=temperature,
            system_message=system_message,
            response_format=response_format,
            **kwargs
        )
        
        # Return content and usage
        return {
            "content": content,
            "usage_metadata": self._last_usage_metadata if self._last_usage_metadata else UsageMetadata()
        }
    
    def get_last_usage_metadata(self) -> Optional[UsageMetadata]:
        """
        Get the usage metadata from the last LLM call.
        
        Returns:
            UsageMetadata object or None if not available
        """
        return self._last_usage_metadata
