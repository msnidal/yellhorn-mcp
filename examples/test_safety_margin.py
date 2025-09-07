#!/usr/bin/env python3
"""Example demonstrating the percentage-based safety margin in LLM Manager."""

import asyncio

from yellhorn_mcp.llm import LLMManager
from yellhorn_mcp.utils.token_utils import TokenCounter


async def main():
    """Demonstrate safety margin calculation for different models."""
    
    # Create LLM Manager with 10% safety margin (default)
    llm_manager = LLMManager()
    token_counter = TokenCounter()
    
    # Test models with different limits
    models = [
        "gpt-4o",           # 128k limit
        "gpt-4o-mini",      # 128k limit  
        "gemini-2.0-flash-exp",  # 1M limit
        "gemini-1.5-pro",   # 2M limit
    ]
    
    print("Percentage-based Safety Margin Calculation (10% of model limit):")
    print("=" * 70)
    
    for model in models:
        model_limit = token_counter.get_model_limit(model)
        safety_margin = int(model_limit * llm_manager.safety_margin_ratio)
        available_tokens = model_limit - safety_margin
        
        print(f"\nModel: {model}")
        print(f"  Token Limit: {model_limit:,}")
        print(f"  Safety Margin (10%): {safety_margin:,}")
        print(f"  Available for Input: {available_tokens:,}")
    
    print("\n" + "=" * 70)
    
    # Example with custom safety margin
    custom_llm_manager = LLMManager(config={"safety_margin_ratio": 0.15})  # 15%
    
    print("\nCustom Safety Margin (15%):")
    print("-" * 40)
    
    model = "gemini-2.0-flash-exp"
    model_limit = token_counter.get_model_limit(model)
    safety_margin = int(model_limit * custom_llm_manager.safety_margin_ratio)
    available_tokens = model_limit - safety_margin
    
    print(f"Model: {model}")
    print(f"  Token Limit: {model_limit:,}")
    print(f"  Safety Margin (15%): {safety_margin:,}")
    print(f"  Available for Input: {available_tokens:,}")
    
    print("\n" + "=" * 70)
    print("\nBenefits of percentage-based safety margin:")
    print("1. Scales automatically with model capacity")
    print("2. Larger models get proportionally larger buffers")
    print("3. Consistent behavior across different model sizes")
    print("4. Configurable via safety_margin_ratio parameter")


if __name__ == "__main__":
    asyncio.run(main())
