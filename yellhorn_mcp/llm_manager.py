"""Unified LLM Manager with automatic chunking support and rate limit handling."""

import asyncio
import json
import re
import time
from typing import Any, Dict, List, Optional, Union
from openai import AsyncOpenAI, RateLimitError
from google import genai
from google.api_core import exceptions as google_exceptions
from tenacity import (
    retry,
    retry_if_exception,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
    before_sleep_log,
    RetryCallState
)
import logging
from .token_counter import TokenCounter

# Configure logging
logger = logging.getLogger(__name__)

def log_retry_attempt(retry_state: RetryCallState) -> None:
    """Log the retry attempt with exponential backoff details."""
    if retry_state.outcome is None:
        return
        
    attempt = retry_state.attempt_number
    wait_time = retry_state.outcome_timestamp - retry_state.start_time
    
    logger.warning(
        f"Retrying {retry_state.fn.__name__} after {wait_time:.1f} seconds "
        f"(attempt {attempt}): {str(retry_state.outcome.exception())}"
    )

def is_retryable_error(exception: Exception) -> bool:
    """Check if the exception is retryable."""
    # Handle ClientError from google.generativeai which wraps the actual error
    if hasattr(exception, 'message') and hasattr(exception, 'code'):
        error_message = str(exception.message).lower()
        error_code = getattr(exception, 'code', None)
        
        # Check for rate limiting or quota exceeded
        if error_code == 429 or 'resource_exhausted' in error_message or 'quota' in error_message:
            return True
    
    # Check for standard retryable exceptions
    if any(isinstance(exception, exc_type) for exc_type in [
        RateLimitError,
        google_exceptions.ResourceExhausted,
        google_exceptions.TooManyRequests,
        ConnectionError,
        asyncio.TimeoutError
    ]):
        return True
    
    # Check for error messages in string representation
    error_message = str(exception).lower()
    if any(term in error_message for term in ['resource_exhausted', 'quota', 'rate limit', 'too many requests']):
        return True
        
    return False

# Common retry decorator for API calls
api_retry = retry(
    retry=retry_if_exception(is_retryable_error),
    wait=wait_exponential(
        multiplier=1,
        min=4,
        max=60,
        exp_base=2
    ),
    stop=stop_after_attempt(5),
    before_sleep=log_retry_attempt,
    reraise=True
)


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
        try:
            return self.total_tokens is not None and self.total_tokens > 0
        except (TypeError, AttributeError):
            return False

class ChunkingStrategy:
    """Strategies for splitting text into chunks while respecting token limits and natural boundaries."""
    
    @staticmethod
    def _find_split_point(text: str, max_length: int) -> int:
        """
        Find the best split point in text before max_length.
        Prefers splitting at paragraph breaks, then at sentence boundaries, then at word boundaries.
        """
        # First try to split at paragraph breaks
        para_break = text.rfind('\n', 0, max_length)
        if para_break > 0:
            return para_break + 2  # Include the newlines
            
        # Then try to split at sentence boundaries
        sentence_break = max(
            text.rfind('. ', 0, max_length),
            text.rfind('! ', 0, max_length),
            text.rfind('? ', 0, max_length),
            text.rfind('\n', 0, max_length)  # Or at least at a newline
        )
        
        if sentence_break > 0:
            return sentence_break + 1  # Include the space or newline
            
        # Finally, split at the last space before max_length
        space_break = text.rfind(' ', 0, max_length)
        if space_break > 0:
            return space_break
            
        # If no good break found, split at max_length
        return max_length
    
    @staticmethod
    def split_by_sentences(
        text: str, 
        max_tokens: int, 
        token_counter: TokenCounter,
        model: str,
        overlap_ratio: float = 0.1,
        safety_margin_tokens: int = 50
    ) -> List[str]:
        """
        Split text into chunks that don't exceed max_tokens, trying to respect sentence boundaries.
        
        Args:
            text: Text to split
            max_tokens: Maximum tokens per chunk
            token_counter: TokenCounter instance
            model: Model name for token counting
            overlap_ratio: Ratio of overlap between chunks (0.0 to 0.5)
            
        Returns:
            List of text chunks
        """
        if not text.strip():
            return []
        
        target_tokens = max_tokens - safety_margin_tokens
        chunks = []
        remaining_text = text
        overlap_tokens = int(max_tokens * overlap_ratio)
        
        while remaining_text:
            # Estimate chunk size
            estimated_chars = len(remaining_text)
            estimated_tokens = token_counter.count_tokens(remaining_text[:estimated_chars], model)
            
            # Adjust chunk size based on token count
            if estimated_tokens > target_tokens:
                # Binary search for the right split point
                low = 0
                high = len(remaining_text)
                best_split = len(remaining_text)
                
                while low <= high:
                    mid = (low + high) // 2
                    chunk = remaining_text[:mid]
                    tokens = token_counter.count_tokens(chunk, model)
                    
                    if tokens <= target_tokens:
                        best_split = mid
                        low = mid + 1
                    else:
                        high = mid - 1
                
                # Find the best split point near the token limit
                if best_split < len(remaining_text):
                    split_pos = ChunkingStrategy._find_split_point(
                        remaining_text[:best_split], best_split
                    )
                    # Ensure we make progress
                    if split_pos == 0 or split_pos == best_split:
                        split_pos = best_split
                else:
                    split_pos = best_split
            else:
                split_pos = len(remaining_text)
            
            # Extract the chunk and remaining text
            chunk = remaining_text[:split_pos].strip()
            remaining_text = remaining_text[split_pos:].strip()
            
            if not chunk:
                break
                
            chunks.append(chunk)
            
            # Add overlap if there's more text to process
            if remaining_text and overlap_tokens > 0:
                # Find the start of the next sentence for overlap
                next_sentence_start = 0
                for i, c in enumerate(remaining_text):
                    if c in '.!?':
                        next_sentence_start = i + 1
                        if next_sentence_start < len(remaining_text) and remaining_text[next_sentence_start] == ' ':
                            next_sentence_start += 1
                        break
                
                if next_sentence_start > 0 and next_sentence_start < len(remaining_text):
                    overlap_text = remaining_text[:next_sentence_start]
                    remaining_text = overlap_text + remaining_text[next_sentence_start:]
        
        return chunks
    
    @staticmethod
    def split_by_paragraphs(
        text: str,
        max_tokens: int,
        token_counter: TokenCounter,
        model: str,
        overlap_ratio: float = 0.1,
        safety_margin_tokens: int = 50
    ) -> List[str]:
        """
        Split text into chunks by paragraphs, respecting token limits.
        
        Args:
            text: Text to split
            max_tokens: Maximum tokens per chunk
            token_counter: TokenCounter instance
            model: Model name for token counting
            overlap_ratio: Ratio of overlap between chunks (0.0 to 0.5)
            
        Returns:
            List of text chunks
        """
        if not text.strip():
            return []
            
        target_tokens = max_tokens - safety_margin_tokens
        # First split by paragraphs
        paragraphs = [p for p in text.split('\n') if p.strip()]
        chunks = []
        current_chunk = []
        current_tokens = 0
        
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
                
            para_tokens = token_counter.count_tokens(para, model)
            
            # If paragraph is too large, split it by sentences
            if para_tokens > target_tokens:
                # Flush current chunk if not empty
                if current_chunk:
                    chunks.append('\n'.join(current_chunk))
                    current_chunk = []
                    current_tokens = 0
                
                # Split the large paragraph by sentences
                chunks.extend(
                    ChunkingStrategy.split_by_sentences(
                        para, max_tokens, token_counter, model, overlap_ratio, safety_margin_tokens
                    )
                )
            # If adding this paragraph would exceed the token limit
            elif current_tokens + para_tokens > target_tokens and current_chunk:
                chunks.append('\n'.join(current_chunk))
                
                # Add overlap from previous chunk if needed
                if overlap_ratio > 0 and chunks:
                    overlap_tokens = int(max_tokens * overlap_ratio)
                    overlap_text = '\n'.join(current_chunk)
                    overlap_text = overlap_text[-overlap_tokens * 4:]  # Rough estimate of chars per token
                    current_chunk = [overlap_text, para]
                    current_tokens = token_counter.count_tokens('\n'.join(current_chunk), model)
                else:
                    current_chunk = [para]
                    current_tokens = para_tokens
            else:
                current_chunk.append(para)
                current_tokens += para_tokens + 2  # Account for newlines
        
        # Add the last chunk if not empty
        if current_chunk:
            chunks.append('\n'.join(current_chunk))
        
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
        self.token_counter = TokenCounter(config)
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
    
    @api_retry
    async def _call_openai(
        self,
        prompt: str,
        model: str,
        temperature: float,
        system_message: Optional[str],
        response_format: Optional[str],
        **kwargs
    ) -> Union[str, Dict[str, Any]]:
        """Call OpenAI API with automatic retry on rate limits.
        
        Args:
            prompt: The prompt to send
            model: Model name
            temperature: Temperature for generation
            system_message: Optional system message
            response_format: Optional response format (e.g., "json")
            **kwargs: Additional parameters for the OpenAI API
            
        Returns:
            Generated response (string or dict if JSON format)
            
        Raises:
            RateLimitError: If rate limited and max retries exceeded
            ValueError: If OpenAI client is not initialized
        """
        if not self.openai_client:
            raise ValueError("OpenAI client not initialized")
        
        messages = []
        if system_message:
            messages.append({"role": "system", "content": system_message})
        messages.append({"role": "user", "content": prompt})
        
        params = {
            "model": model,
            "messages": messages,
            "temperature": 1.0 if model.startswith("o") else temperature,
            **kwargs
        }
        
        if response_format == "json":
            params["response_format"] = {"type": "json_object"}
        
        try:
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

        except Exception as e:
            logger.error(f"OpenAI API call failed: {str(e)}")
            raise
    
    @api_retry
    async def _call_gemini(
        self,
        prompt: str,
        model: str,
        temperature: float,
        system_message: Optional[str],
        response_format: Optional[str],
        **kwargs
    ) -> Union[str, Dict[str, Any]]:
        """Call Gemini API with automatic retry on rate limits.
        
        Args:
            prompt: The prompt to send
            model: Model name
            temperature: Temperature for generation
            system_message: Optional system message
            response_format: Optional response format (e.g., "json")
            **kwargs: Additional parameters for the Gemini API
            
        Returns:
            Generated response (string or dict if JSON format)
            
        Raises:
            google.api_core.exceptions.ResourceExhausted: If rate limited and max retries exceeded
            ValueError: If Gemini client is not configured
        """
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
        
        # Extract generation_config if provided (for search grounding)
        generation_config = kwargs.pop("generation_config", None)
        
        # Build config
        config_dict = {
            "temperature": temperature,
            "response_mime_type": "application/json" if response_format == "json" else "text/plain",
        }
        
        # Add any additional kwargs (excluding generation_config)
        config_dict.update(kwargs)
        
        # Create config instance
        if config_class == GenerateContentConfig:
            config = config_class(**config_dict)
        else:
            config = config_dict
        
        try:
            # Prepare API call parameters
            api_params = {
                "model": f"models/{model}",
                "contents": full_prompt,
                "config": config
            }
            
            # Add generation_config if provided (for search grounding)
            if generation_config:
                api_params["config"] = generation_config
            
            # Make the API call
            response = await self.gemini_client.aio.models.generate_content(**api_params)
            
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
                    try:
                        return json.loads(json_matches[0])
                    except json.JSONDecodeError:
                        return {"error": "No valid JSON found in response", "content": content}
                else:
                    return {"error": "No JSON content found in response"}
            
            # Store the response object for potential citation processing
            if hasattr(response, 'grounding_metadata'):
                # Store metadata in a thread-local or instance variable for retrieval
                self._last_gemini_response = response
            
            return content
            
        except Exception as e:
            logger.error(f"Gemini API call failed: {str(e)}")
            raise
    
    async def _chunked_call(
        self,
        prompt: str,
        model: str,
        temperature: float,
        system_message: Optional[str],
        response_format: Optional[str],
        **kwargs
    ) -> Union[str, Dict[str, Any]]:
        """Make chunked LLM calls and aggregate results with rate limit handling.
        
        Args:
            prompt: The prompt to send
            model: Model name
            temperature: Temperature for generation
            system_message: Optional system message
            response_format: Optional response format (e.g., "json")
            **kwargs: Additional parameters for the LLM API
            
        Returns:
            Aggregated response (string or dict if JSON format)
        """
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
