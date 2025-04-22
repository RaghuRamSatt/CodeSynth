# Install Google's genai Python client first
# pip install google-genai

import os
import logging
from typing import Dict, Any, Optional, List

from google import genai
from google.genai import types

from agents.base_agent import BaseAgent
from utils.prompt_templates import GEMINI_PROMPT_TEMPLATES

logger = logging.getLogger(__name__)

class GeminiAgent(BaseAgent):
    """
    Agent implementation using Google's Gemini 2.5 Flash
    """
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """Initialize the Gemini agent."""
        super().__init__(config_path)
        self.client = None
        self.model_name = self.config.get("models", {}).get("gemini", {}).get("model_name", "gemini-2.5-flash-preview-04-17")
        self.max_tokens = self.config.get("models", {}).get("gemini", {}).get("max_tokens", 4096)
        self.temperature = self.config.get("models", {}).get("gemini", {}).get("temperature", 1.0)
        self.thinking_budget = self.config.get("models", {}).get("gemini", {}).get("thinking_budget", 0)
        self.api_key = os.getenv("GEMINI_API_KEY", "")
        self.generation_config = types.GenerateContentConfig(
        max_output_tokens=self.max_tokens,
        temperature=self.temperature,
        thinking_config=types.ThinkingConfig(thinking_budget=self.thinking_budget)
        )
        self.prompt_templates = GEMINI_PROMPT_TEMPLATES
        
    def initialize(self) -> bool:
        """Initialize the Gemini client."""
        try:
            # Try loading from .env file directly
            from dotenv import load_dotenv
            load_dotenv()
            
            self.api_key = os.getenv("GEMINI_API_KEY", "")
            
            # Add debug logging
            logger.info(f"API key found: {'Yes' if self.api_key else 'No'}")
            
            if not self.api_key:
                logger.error("GEMINI_API_KEY environment variable not set")
                return False
                    
            self.client = genai.Client(api_key=self.api_key)
            
            # Test connection with a simple prompt
            test_response = self.client.models.generate_content(
                model=self.model_name,
                config=self.generation_config,
                contents="Hello, Gemini!"
            )
            logger.info(f"Gemini agent initialized successfully with model: {self.model_name}")
            print("Response from Gemini agent initialization - ", test_response.candidates[0].content)
            return True
        except Exception as e:
            logger.error(f"Failed to initialize Gemini agent: {e}")
            return False
            
    def generate_code(self, prompt: str, dataset_info: Dict[str, Any]) -> str:
        """Generate Python code using Gemini."""
        if not self.client:
            if not self.initialize():
                return "Error: Gemini agent is not initialized. Please check your API key."
        
        formatted_dataset_info = self.format_dataset_info(dataset_info)
        full_prompt = self.prompt_templates["code_generation"].format(
            user_prompt=prompt,
            dataset_info=formatted_dataset_info
        )
        
        try:
            config = self.generation_config
            config.system_instruction = "You are a senior data scientist specialized in Python programming for data analysis. Your task is to generate high-quality, efficient, and well-documented Python code to address user questions about their dataset. Focus on creating code that is robust, handles errors gracefully, and produces insightful results. Always include explanatory comments. Only include code in your response, no additional explanations."
            response = self.client.models.generate_content(
                model=self.model_name,
                config=config,
                contents=full_prompt
            )
            
            # Extract code from the response
            code = self._extract_code_from_response(response.candidates[0].content.parts)
            return code
        except Exception as e:
            logger.error(f"Error generating code with Gemini: {e}")
            return f"Error generating code: {str(e)}"
            
    def answer_question(self, question: str, context: Optional[str] = None) -> str:
        """Answer a question about code or dataset."""
        if not self.client:
            if not self.initialize():
                return "Error: Gemini agent is not initialized. Please check your API key."
        
        full_prompt = self.prompt_templates["question_answering"].format(
            user_question=question,
            context=context or ""
        )
        
        try:
            config = self.generation_config
            config.system_instruction = "You are a helpful data science assistant that provides clear and accurate answers to questions about code, data analysis, and statistics. Provide explanations that are accessible but technically precise."
            response = self.client.models.generate_content(
                model=self.model_name,
                config=config,
                contents=full_prompt
            )
            
            return response.candidates[0].content.parts
        except Exception as e:
            logger.error(f"Error answering question with Gemini: {e}")
            return f"Error answering question: {str(e)}"
            
    def improve_code(self, code: str, feedback: str) -> str:
        """Improve existing code based on user feedback."""
        if not self.client:
            if not self.initialize():
                return "Error: Gemini agent is not initialized. Please check your API key."
        
        full_prompt = self.prompt_templates["code_improvement"].format(
            original_code=code,
            user_feedback=feedback
        )
        
        try:
            config = self.generation_config
            config.system_instruction = "You are a Python code optimization expert. Your task is to improve existing Python code based on user feedback while maintaining its core functionality. Focus on clarity, efficiency, and best practices. Only include the improved code in your response, no additional explanations."
            response = self.client.models.generate_config(
                model=self.model_name,
                config=config,
                contents=full_prompt
            )
            
            # Extract code from the response
            improved_code = self._extract_code_from_response(response.candidates[0].content.parts)
            return improved_code
        except Exception as e:
            logger.error(f"Error improving code with Gemini: {e}")
            return f"Error improving code: {str(e)}"
    
    def _extract_code_from_response(self, content) -> str:
        """Extract Python code from Gemini's response."""
        code = ""
        for block in content:
            if block.text is not None:
                text = block.text
                # Try to extract code blocks with ```python ... ``` format
                if "```python" in text:
                    code_blocks = text.split("```python")
                    for block in code_blocks[1:]:  # Skip the first part before ```python
                        if "```" in block:
                            code += block.split("```")[0].strip() + "\n\n"
                # If no python blocks found, look for any code blocks
                elif "```" in text and not code:
                    code_blocks = text.split("```")
                    # Take the content between ``` markers (odd indices)
                    for i in range(1, len(code_blocks), 2):
                        if i < len(code_blocks):
                            code += code_blocks[i].strip() + "\n\n"
                
        # If no code blocks found, check if the text appears to be code itself
        if not code:
            for block in content:
                if block.data == "text" and ("import " in block.text or "def " in block.text or "class " in block.text):
                    code = block.text
                    
        return code.strip()