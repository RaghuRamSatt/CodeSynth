"""
OpenAI Agent - Implementation for GPT-4 models
"""

import os
import logging
from typing import Dict, Any, Optional
import json

from openai import OpenAI
from agents.base_agent import BaseAgent

logger = logging.getLogger(__name__)

class OpenAIAgent(BaseAgent):
    """
    Agent implementation using OpenAI's GPT-4 models.
    """
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """Initialize the OpenAI agent."""
        super().__init__(config_path)
        self.client = None
        self.model_name = self.config.get("models", {}).get("openai", {}).get("model_name", "gpt-4.1")
        self.max_tokens = self.config.get("models", {}).get("openai", {}).get("max_tokens", 4096)
        self.temperature = self.config.get("models", {}).get("openai", {}).get("temperature", 0.2)
        self.api_key = os.getenv("OPENAI_API_KEY", "")
        self.last_prompt_tokens = 0
        self.last_completion_tokens = 0
        
        # Define prompt templates
        self.prompt_templates = {
            "code_generation": (
                "Generate Python code for the following data analysis task:\n\n"
                "User query: {user_prompt}\n\n"
                "Dataset information:\n{dataset_info}\n\n"
                "Requirements:\n"
                "1. The dataset is already loaded as a pandas DataFrame named 'df'\n"
                "2. Use pandas, numpy, matplotlib, seaborn, and scikit-learn as needed\n"
                "3. Include visualizations when appropriate\n"
                "4. Handle potential errors and edge cases\n"
                "5. Include clear comments explaining your approach\n"
                "6. Ensure the code is efficient and follows best practices\n\n"
                "Please respond with well-documented Python code that addresses the query."
            ),
            "question_answering": (
                "Please answer the following question about data analysis or Python code:\n\n"
                "Question: {user_question}\n\n"
                "Context (if relevant):\n{context}\n\n"
                "Provide a clear, accurate, and detailed answer."
            ),
            "code_improvement": (
                "I have some Python code for data analysis that needs improvement:\n\n"
                "```python\n{original_code}\n```\n\n"
                "User feedback: {user_feedback}\n\n"
                "Please provide an improved version of this code that addresses the feedback."
                "Make the code more robust, efficient, and well-documented where needed."
            )
        }
        
    def initialize(self) -> bool:
        """Initialize the OpenAI client."""
        try:
            if not self.api_key:
                logger.warning("OpenAI API key not found")
                return False
                
            self.client = OpenAI(api_key=self.api_key)
            logger.info(f"OpenAI agent initialized successfully with model: {self.model_name}")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize OpenAI agent: {e}")
            return False
            
    def generate_code(self, prompt: str, dataset_info: Dict[str, Any]) -> str:
        """Generate Python code using OpenAI."""
        if not self.client:
            if not self.initialize():
                return "Error: OpenAI agent is not initialized. Please check your API key."
        
        formatted_dataset_info = self.format_dataset_info(dataset_info)
        full_prompt = self.prompt_templates["code_generation"].format(
            user_prompt=prompt,
            dataset_info=formatted_dataset_info
        )
        
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {
                        "role": "system", 
                        "content": "You are an expert data scientist. Generate clear, efficient, and well-documented Python code for data analysis tasks."
                    },
                    {
                        "role": "user", 
                        "content": full_prompt
                    }
                ],
                max_tokens=self.max_tokens,
                temperature=self.temperature
            )
            
            # Store token usage
            self.last_prompt_tokens = response.usage.prompt_tokens
            self.last_completion_tokens = response.usage.completion_tokens
            
            # Extract code from the response
            code = self._extract_code_from_response(response.choices[0].message.content)
            
            # Format output with prefix and code
            prefix = "Here's an analysis approach for your data:"
            if "```" not in response.choices[0].message.content:
                # No code blocks, return the whole response
                return {"prefix": prefix, "code": code}
            
            # Process any text before the first code block as the prefix
            content = response.choices[0].message.content
            if "```python" in content:
                parts = content.split("```python", 1)
                if parts[0].strip():
                    prefix = parts[0].strip()
            elif "```" in content:
                parts = content.split("```", 1)
                if parts[0].strip():
                    prefix = parts[0].strip()
                    
            return {"prefix": prefix, "code": code}
            
        except Exception as e:
            logger.error(f"Error generating code with OpenAI: {e}")
            return f"Error generating code: {str(e)}"

    def answer_question(self, question: str, context: Optional[str] = None) -> str:
        """Answer a question about code or dataset."""
        if not self.client:
            if not self.initialize():
                return "Error: OpenAI agent is not initialized. Please check your API key."
        
        full_prompt = self.prompt_templates["question_answering"].format(
            user_question=question,
            context=context or ""
        )
        
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {
                        "role": "system", 
                        "content": "You are a helpful data science expert providing clear and accurate answers."
                    },
                    {
                        "role": "user", 
                        "content": full_prompt
                    }
                ],
                max_tokens=self.max_tokens,
                temperature=self.temperature
            )
            
            # Store token usage
            self.last_prompt_tokens = response.usage.prompt_tokens
            self.last_completion_tokens = response.usage.completion_tokens
            
            return response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"Error answering question with OpenAI: {e}")
            return f"Error answering question: {str(e)}"

    def improve_code(self, code: str, feedback: str) -> str:
        """Improve code based on feedback."""
        if not self.client:
            if not self.initialize():
                return "Error: OpenAI agent is not initialized. Please check your API key."
        
        full_prompt = self.prompt_templates["code_improvement"].format(
            original_code=code,
            user_feedback=feedback
        )
        
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {
                        "role": "system", 
                        "content": "You are an expert Python programmer specializing in improving data science code."
                    },
                    {
                        "role": "user", 
                        "content": full_prompt
                    }
                ],
                max_tokens=self.max_tokens,
                temperature=self.temperature
            )
            
            # Store token usage
            self.last_prompt_tokens = response.usage.prompt_tokens
            self.last_completion_tokens = response.usage.completion_tokens
            
            return self._extract_code_from_response(response.choices[0].message.content)
            
        except Exception as e:
            logger.error(f"Error improving code with OpenAI: {e}")
            return f"Error improving code: {str(e)}"
        
    def _extract_code_from_response(self, response_content: str) -> str:
        """Extract Python code from OpenAI's response."""
        code = ""
        
        # Try to extract code blocks with ```python ... ``` format
        if "```python" in response_content:
            code_blocks = response_content.split("```python")
            for block in code_blocks[1:]:  # Skip the first part before ```python
                if "```" in block:
                    code += block.split("```")[0].strip() + "\n\n"
        # If no python blocks found, look for any code blocks
        elif "```" in response_content and not code:
            code_blocks = response_content.split("```")
            # Take the content between ``` markers (odd indices)
            for i in range(1, len(code_blocks), 2):
                if i < len(code_blocks):
                    code += code_blocks[i].strip() + "\n\n"
        
        # If no code blocks found, check if the text appears to be code itself
        if not code and ("import " in response_content or "def " in response_content or "class " in response_content):
            code = response_content
                    
        return code.strip()