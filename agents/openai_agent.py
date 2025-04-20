# Install Anthropic's Python client first
# pip install anthropic

import os
import logging
from typing import Dict, Any, Optional, List

# import anthropic
from anthropic.types import ContentBlockParam
from typing import Union
import openai

from agents.base_agent import BaseAgent
from utils.prompt_templates import OPENAI_PROMPT_TEMPLATES

logger = logging.getLogger(__name__)

class OpenAIAgent(BaseAgent):
    """
    Agent implementation using Open AI's GPT 4.1 and/or GPT o4-mini
    """
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """Initialize the Open AI agent."""
        super().__init__(config_path)
        self.client = None
        self.model_name = self.config.get("models", {}).get("openai", {}).get("model_name", "o4-mini")
        self.max_tokens = self.config.get("models", {}).get("openai", {}).get("max_tokens", 1024)
        self.temperature = self.config.get("models", {}).get("openai", {}).get("temperature", 0)
        self.api_key = os.getenv("OPENAI_API_KEY", "")
        self.prompt_templates = OPENAI_PROMPT_TEMPLATES
        
    def initialize(self) -> bool:
        """Initialize the Open AI client."""
        try:
            # Try loading from .env file directly
            from dotenv import load_dotenv
            load_dotenv()
            
            self.api_key = os.getenv("OPENAI_API_KEY", "")
            
            # Add debug logging
            logger.info(f"API key found: {'Yes' if self.api_key else 'No'}")
            
            if not self.api_key:
                logger.error("OPENAI_API_KEY environment variable not set")
                return False
                    
            self.client = openai.OpenAI(api_key=self.api_key)
            # Test connection with a simple prompt
            test_response = self.client.chat.completions.create(
                model=self.model_name,
                max_tokens=10,
                messages=[{"role": "user", "content": "Hello"}]
            )
            logger.info(f"Open AI agent initialized successfully with model: {self.model_name}")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize Open AI agent: {e}")
            return False
            
    def generate_code(self, prompt: str, dataset_info: Dict[str, Any]) -> str:
        """Generate Python code using Open AI."""
        if not self.client:
            if not self.initialize():
                return "Error: Open AI agent is not initialized. Please check your API key."
        
        formatted_dataset_info = self.format_dataset_info(dataset_info)
        full_prompt = self.prompt_templates["code_generation"].format(
            user_prompt=prompt,
            dataset_info=formatted_dataset_info
        )
        
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                messages=[
                    {"role": "system", "content" : "You are a senior data scientist specialized in Python programming for data analysis. Your task is to generate high-quality, efficient, and well-documented Python code to address user questions about their dataset. Focus on creating code that is robust, handles errors gracefully, and produces insightful results. Always include explanatory comments. Only include code in your response, no additional explanations."},
                    {"role": "user", "content": full_prompt}
                    ]
            )
            
            # Extract code from the response
            code = self._extract_code_from_response(response.choices[0].message.content)
            return code
        except Exception as e:
            logger.error(f"Error generating code with Open AI: {e}")
            return f"Error generating code: {str(e)}"
            
    def answer_question(self, question: str, context: Optional[str] = None) -> str:
        """Answer a question about code or dataset."""
        if not self.client:
            if not self.initialize():
                return "Error: Open AI agent is not initialized. Please check your API key."
        
        full_prompt = self.prompt_templates["question_answering"].format(
            user_question=question,
            context=context or ""
        )
        
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                messages=[
                    {"role": "system", "content" : "You are a senior data scientist specialized in Python programming for data analysis. Your task is to generate high-quality, efficient, and well-documented Python code to address user questions about their dataset. Focus on creating code that is robust, handles errors gracefully, and produces insightful results. Always include explanatory comments. Only include code in your response, no additional explanations."},
                    {"role": "user", "content": full_prompt}
                    ]
            )
            
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"Error answering question with Open AI: {e}")
            return f"Error answering question: {str(e)}"
            
    def improve_code(self, code: str, feedback: str) -> str:
        """Improve existing code based on user feedback."""
        if not self.client:
            if not self.initialize():
                return "Error: Open AI agent is not initialized. Please check your API key."
        
        full_prompt = self.prompt_templates["code_improvement"].format(
            original_code=code,
            user_feedback=feedback
        )
        
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                messages=[
                    {"role": "system", "content" : "You are a senior data scientist specialized in Python programming for data analysis. Your task is to generate high-quality, efficient, and well-documented Python code to address user questions about their dataset. Focus on creating code that is robust, handles errors gracefully, and produces insightful results. Always include explanatory comments. Only include code in your response, no additional explanations."},
                    {"role": "user", "content": full_prompt}
                    ]
            )
            
            # Extract code from the response
            improved_code = self._extract_code_from_response(response.choices[0].message.content)
            return improved_code
        except Exception as e:
            logger.error(f"Error improving code with Open A: {e}")
            return f"Error improving code: {str(e)}"
    
    def _extract_code_from_response(self, content: Union[str, List[ContentBlockParam]]) -> str:
        """Extract Python code from OpenAI or Anthropic response content."""
        code = ""
        
        # Normalize input to a list of strings
        if isinstance(content, str):
            texts = [content]
        elif isinstance(content, list):
            texts = [block.text for block in content if getattr(block, "type", None) == "text"]
        else:
            raise ValueError("Unsupported content format")

        # Extract code blocks
        for text in texts:
            if "```python" in text:
                code_blocks = text.split("```python")
                for block in code_blocks[1:]:  # Skip the first part before ```python
                    if "```" in block:
                        code += block.split("```")[0].strip() + "\n\n"
            elif "```" in text and not code:  # fallback for generic code blocks
                code_blocks = text.split("```")
                for i in range(1, len(code_blocks), 2):  # only take odd indices (code)
                    code += code_blocks[i].strip() + "\n\n"
        
        return code.strip()
                
        # If no code blocks found, check if the text appears to be code itself
        if not code:
            for block in content:
                if block.type == "text" and ("import " in block.text or "def " in block.text or "class " in block.text):
                    code = block.text
                    
        return code.strip()