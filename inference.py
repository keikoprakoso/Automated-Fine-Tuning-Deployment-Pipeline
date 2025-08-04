"""
Model Inference Module for LLM Pipeline

This module handles inference with fine-tuned models, token usage tracking,
and response generation. Provides cost estimation and logging capabilities.

Author: Keiko Rafi Ananda Prakoso
Date: 2025
"""

import os
import json
import logging
import csv
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import openai
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class InferenceManager:
    """
    Manages model inference with fine-tuned models.
    
    Handles prompt processing, response generation, token usage tracking,
    and cost estimation. Logs all interactions for monitoring.
    """
    
    def __init__(self, api_key: Optional[str] = None, log_file: str = "logs/inference_logs.csv"):
        """
        Initialize the InferenceManager.
        
        Args:
            api_key (str, optional): OpenAI API key. If None, reads from environment.
            log_file (str): Path to CSV log file for usage tracking.
        """
        self.api_key = api_key or os.getenv('OPENAI_API_KEY')
        if not self.api_key:
            raise ValueError("OpenAI API key is required. Set OPENAI_API_KEY environment variable.")
        
        openai.api_key = self.api_key
        self.log_file = log_file
        self._init_log_file()
        
        # Token pricing (as of 2024, may need updates)
        self.token_pricing = {
            "gpt-3.5-turbo": {"input": 0.0015, "output": 0.002},  # per 1K tokens
            "gpt-4": {"input": 0.03, "output": 0.06},  # per 1K tokens
            "gpt-4-turbo": {"input": 0.01, "output": 0.03},  # per 1K tokens
        }
    
    def _init_log_file(self):
        """Initialize the log file with headers if it doesn't exist."""
        try:
            log_path = Path(self.log_file)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            
            if not log_path.exists():
                with open(log_path, 'w', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        'timestamp',
                        'model_id',
                        'prompt',
                        'response',
                        'prompt_tokens',
                        'completion_tokens',
                        'total_tokens',
                        'estimated_cost_usd',
                        'response_time_seconds',
                        'status',
                        'error_message'
                    ])
                logger.info(f"Initialized log file: {self.log_file}")
        except Exception as e:
            logger.error(f"Failed to initialize log file: {str(e)}")
    
    def estimate_tokens(self, text: str) -> int:
        """
        Estimate token count for a given text.
        
        This is a rough estimation. For precise counts, use OpenAI's tokenizer.
        
        Args:
            text (str): Text to estimate tokens for
            
        Returns:
            int: Estimated token count
        """
        # Rough estimation: 1 token ≈ 4 characters for English text
        return len(text) // 4
    
    def estimate_cost(self, model: str, prompt_tokens: int, completion_tokens: int) -> float:
        """
        Estimate cost for a model inference call.
        
        Args:
            model (str): Model name
            prompt_tokens (int): Number of input tokens
            completion_tokens (int): Number of output tokens
            
        Returns:
            float: Estimated cost in USD
        """
        # Find base model for pricing
        base_model = None
        for base in self.token_pricing.keys():
            if base in model:
                base_model = base
                break
        
        if not base_model:
            # Default to gpt-3.5-turbo pricing for fine-tuned models
            base_model = "gpt-3.5-turbo"
        
        pricing = self.token_pricing[base_model]
        
        # Calculate cost (pricing is per 1K tokens)
        input_cost = (prompt_tokens / 1000) * pricing["input"]
        output_cost = (completion_tokens / 1000) * pricing["output"]
        
        return input_cost + output_cost
    
    def generate_response(self, 
                         prompt: str, 
                         model_id: str,
                         max_tokens: int = 1000,
                         temperature: float = 0.7,
                         system_message: Optional[str] = None) -> Dict:
        """
        Generate response using fine-tuned model.
        
        Args:
            prompt (str): User prompt
            model_id (str): Fine-tuned model ID
            max_tokens (int): Maximum tokens for response
            temperature (float): Response randomness (0.0 to 2.0)
            system_message (str, optional): System message for context
            
        Returns:
            Dict: Response with metadata
        """
        start_time = datetime.now()
        
        try:
            logger.info(f"Generating response with model: {model_id}")
            
            # Prepare messages
            messages = []
            if system_message:
                messages.append({"role": "system", "content": system_message})
            
            messages.append({"role": "user", "content": prompt})
            
            # Make API call
            response = openai.ChatCompletion.create(
                model=model_id,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature
            )
            
            # Extract response data
            response_text = response.choices[0].message.content
            usage = response.usage
            
            # Calculate response time
            response_time = (datetime.now() - start_time).total_seconds()
            
            # Estimate cost
            estimated_cost = self.estimate_cost(
                model_id, 
                usage.prompt_tokens, 
                usage.completion_tokens
            )
            
            # Prepare result
            result = {
                "response": response_text,
                "prompt_tokens": usage.prompt_tokens,
                "completion_tokens": usage.completion_tokens,
                "total_tokens": usage.total_tokens,
                "estimated_cost_usd": estimated_cost,
                "response_time_seconds": response_time,
                "model_id": model_id,
                "status": "success"
            }
            
            # Log the interaction
            self._log_interaction(prompt, response_text, result)
            
            logger.info(f"Response generated successfully. Tokens: {usage.total_tokens}, Cost: ${estimated_cost:.4f}")
            return result
            
        except Exception as e:
            error_msg = str(e)
            logger.error(f"Inference failed: {error_msg}")
            
            # Log error
            error_result = {
                "response": "",
                "prompt_tokens": self.estimate_tokens(prompt),
                "completion_tokens": 0,
                "total_tokens": self.estimate_tokens(prompt),
                "estimated_cost_usd": 0.0,
                "response_time_seconds": (datetime.now() - start_time).total_seconds(),
                "model_id": model_id,
                "status": "error",
                "error_message": error_msg
            }
            
            self._log_interaction(prompt, "", error_result)
            
            return error_result
    
    def _log_interaction(self, prompt: str, response: str, metadata: Dict):
        """
        Log interaction to CSV file.
        
        Args:
            prompt (str): User prompt
            response (str): Model response
            metadata (Dict): Response metadata
        """
        try:
            with open(self.log_file, 'a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow([
                    datetime.now().isoformat(),
                    metadata.get('model_id', ''),
                    prompt[:500],  # Truncate long prompts
                    response[:500],  # Truncate long responses
                    metadata.get('prompt_tokens', 0),
                    metadata.get('completion_tokens', 0),
                    metadata.get('total_tokens', 0),
                    metadata.get('estimated_cost_usd', 0.0),
                    metadata.get('response_time_seconds', 0.0),
                    metadata.get('status', ''),
                    metadata.get('error_message', '')
                ])
        except Exception as e:
            logger.error(f"Failed to log interaction: {str(e)}")
    
    def get_usage_summary(self, days: int = 30) -> Dict:
        """
        Get usage summary from logs.
        
        Args:
            days (int): Number of days to look back
            
        Returns:
            Dict: Usage summary statistics
        """
        try:
            if not os.path.exists(self.log_file):
                return {"error": "Log file not found"}
            
            # Read logs
            logs = []
            with open(self.log_file, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    logs.append(row)
            
            # Filter by date if needed
            if days > 0:
                cutoff_date = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
                cutoff_date = cutoff_date.replace(day=cutoff_date.day - days)
                
                filtered_logs = []
                for log in logs:
                    try:
                        log_date = datetime.fromisoformat(log['timestamp'])
                        if log_date >= cutoff_date:
                            filtered_logs.append(log)
                    except:
                        continue
                logs = filtered_logs
            
            # Calculate statistics
            total_requests = len(logs)
            successful_requests = len([log for log in logs if log['status'] == 'success'])
            failed_requests = total_requests - successful_requests
            
            total_tokens = sum(int(log.get('total_tokens', 0)) for log in logs)
            total_cost = sum(float(log.get('estimated_cost_usd', 0)) for log in logs)
            
            # Group by model
            model_usage = {}
            for log in logs:
                model = log.get('model_id', 'unknown')
                if model not in model_usage:
                    model_usage[model] = {
                        'requests': 0,
                        'tokens': 0,
                        'cost': 0.0
                    }
                
                model_usage[model]['requests'] += 1
                model_usage[model]['tokens'] += int(log.get('total_tokens', 0))
                model_usage[model]['cost'] += float(log.get('estimated_cost_usd', 0))
            
            summary = {
                "period_days": days,
                "total_requests": total_requests,
                "successful_requests": successful_requests,
                "failed_requests": failed_requests,
                "success_rate": (successful_requests / total_requests * 100) if total_requests > 0 else 0,
                "total_tokens": total_tokens,
                "total_cost_usd": total_cost,
                "model_usage": model_usage,
                "average_tokens_per_request": total_tokens / total_requests if total_requests > 0 else 0,
                "average_cost_per_request": total_cost / total_requests if total_requests > 0 else 0
            }
            
            return summary
            
        except Exception as e:
            logger.error(f"Failed to get usage summary: {str(e)}")
            return {"error": str(e)}
    
    def get_recent_logs(self, limit: int = 10) -> List[Dict]:
        """
        Get recent inference logs.
        
        Args:
            limit (int): Maximum number of logs to return
            
        Returns:
            List[Dict]: Recent log entries
        """
        try:
            if not os.path.exists(self.log_file):
                return []
            
            logs = []
            with open(self.log_file, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    logs.append(row)
            
            # Return most recent logs
            return logs[-limit:] if len(logs) > limit else logs
            
        except Exception as e:
            logger.error(f"Failed to get recent logs: {str(e)}")
            return []
    
    def batch_generate(self, 
                      prompts: List[str], 
                      model_id: str,
                      max_tokens: int = 1000,
                      temperature: float = 0.7) -> List[Dict]:
        """
        Generate responses for multiple prompts.
        
        Args:
            prompts (List[str]): List of user prompts
            model_id (str): Fine-tuned model ID
            max_tokens (int): Maximum tokens per response
            temperature (float): Response randomness
            
        Returns:
            List[Dict]: List of responses with metadata
        """
        logger.info(f"Generating batch responses for {len(prompts)} prompts")
        
        results = []
        for i, prompt in enumerate(prompts):
            logger.info(f"Processing prompt {i+1}/{len(prompts)}")
            result = self.generate_response(
                prompt=prompt,
                model_id=model_id,
                max_tokens=max_tokens,
                temperature=temperature
            )
            results.append(result)
        
        logger.info(f"Batch generation completed. {len(results)} responses generated")
        return results
    
    def test_model(self, model_id: str, test_prompts: List[str] = None) -> Dict:
        """
        Test a fine-tuned model with sample prompts.
        
        Args:
            model_id (str): Model ID to test
            test_prompts (List[str], optional): Test prompts. Uses defaults if None.
            
        Returns:
            Dict: Test results summary
        """
        if test_prompts is None:
            test_prompts = [
                "What is machine learning?",
                "Explain neural networks in simple terms.",
                "How does fine-tuning work?"
            ]
        
        logger.info(f"Testing model: {model_id}")
        
        results = []
        total_tokens = 0
        total_cost = 0.0
        success_count = 0
        
        for prompt in test_prompts:
            result = self.generate_response(prompt, model_id)
            results.append({
                "prompt": prompt,
                "response": result["response"],
                "status": result["status"],
                "tokens": result["total_tokens"],
                "cost": result["estimated_cost_usd"]
            })
            
            total_tokens += result["total_tokens"]
            total_cost += result["estimated_cost_usd"]
            if result["status"] == "success":
                success_count += 1
        
        test_summary = {
            "model_id": model_id,
            "test_prompts": len(test_prompts),
            "successful_responses": success_count,
            "success_rate": (success_count / len(test_prompts)) * 100,
            "total_tokens": total_tokens,
            "total_cost_usd": total_cost,
            "average_tokens_per_response": total_tokens / len(test_prompts),
            "results": results
        }
        
        logger.info(f"Model test completed. Success rate: {test_summary['success_rate']:.1f}%")
        return test_summary


def main():
    """Example usage of the InferenceManager."""
    try:
        # Initialize manager
        manager = InferenceManager()
        
        # Example: Generate response with a fine-tuned model
        # Note: Replace with actual fine-tuned model ID
        model_id = "ft:gpt-3.5-turbo:your-org:your-model-name:1234567890"
        
        response = manager.generate_response(
            prompt="What is the capital of France?",
            model_id=model_id,
            max_tokens=100
        )
        
        print("Response:", response["response"])
        print(f"Tokens used: {response['total_tokens']}")
        print(f"Estimated cost: ${response['estimated_cost_usd']:.4f}")
        
        # Get usage summary
        summary = manager.get_usage_summary(days=7)
        print("Weekly usage summary:", summary)
        
    except Exception as e:
        logger.error(f"Example failed: {str(e)}")


if __name__ == "__main__":
    main() 