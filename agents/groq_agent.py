
import os
import logging
from typing import Dict, Any, Optional

from groq import Groq
from agents.base_agent import BaseAgent
from utils.prompt_templates import GROQ_PROMPT_TEMPLATES

logger = logging.getLogger(__name__)

class GroqAgent(BaseAgent):
    """
    Agent implementation using Groq API (supports both Groq models and Llama models)
    """
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """Initialize the Groq agent."""
        super().__init__(config_path)
        self.client = None
        # Default to Llama model if not specified
        self.model_name = self.config.get("models", {}).get("groq", {}).get("model_name", "llama-3.3-70b-versatile")
        self.max_tokens = self.config.get("models", {}).get("groq", {}).get("max_tokens", 4096)
        self.temperature = self.config.get("models", {}).get("groq", {}).get("temperature", 0.2)
        self.api_key = os.getenv("GROQ_API_KEY", "")
        self.prompt_templates = GROQ_PROMPT_TEMPLATES
        
    def initialize(self) -> bool:
        """Initialize the Groq client."""
        try:
            if not self.api_key:
                logger.warning("Groq API key not found")
                return False
                
            self.client = Groq(api_key=self.api_key)
            logger.info(f"Groq agent initialized successfully with model: {self.model_name}")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize Groq agent: {e}")
            return False
            
    def generate_code(self, prompt: str, dataset_info: Dict[str, Any]) -> str:
        """Generate Python code using Groq."""
        if not self.client:
            if not self.initialize():
                return "Error: Groq agent is not initialized. Please check your API key."
        
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
                        "content": "You are a senior data scientist specialized in Python programming for data analysis. Your task is to generate high-quality, efficient Python code to address user questions about their dataset."
                    },
                    {
                        "role": "user", 
                        "content": full_prompt
                    }
                ],
                max_tokens=self.max_tokens,
                temperature=self.temperature
            )
            
            # Extract code from the response
            code = self._extract_code_from_response(response.choices[0].message.content)
            return code
        except Exception as e:
            logger.error(f"Error generating code with Groq: {e}")
            return f"Error generating code: {str(e)}"
        


    def answer_question(self, question: str, context: Optional[str] = None) -> str:
        """Answer a question about code or dataset."""
        if not self.client:
            if not self.initialize():
                return "Error: Groq agent is not initialized. Please check your API key."
        
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
                        "content": "You are a helpful data science assistant that provides clear and accurate answers to questions about code, data analysis, and statistics."
                    },
                    {
                        "role": "user", 
                        "content": full_prompt
                    }
                ],
                max_tokens=self.max_tokens,
                temperature=self.temperature
            )
            
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"Error answering question with Groq: {e}")
            return f"Error answering question: {str(e)}"

    def improve_code(self, code: str, feedback: str) -> str:
        """Improve code based on feedback."""
        if not self.client:
            if not self.initialize():
                return "Error: Groq agent is not initialized. Please check your API key."
        
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
                        "content": "You are an expert Python programmer specializing in data science code improvement."
                    },
                    {
                        "role": "user", 
                        "content": full_prompt
                    }
                ],
                max_tokens=self.max_tokens,
                temperature=self.temperature
            )
            
            return self._extract_code_from_response(response.choices[0].message.content)
        except Exception as e:
            logger.error(f"Error improving code with Groq: {e}")
            return f"Error improving code: {str(e)}"
        
    def _extract_code_from_response(self, response_content: str) -> str:
        """Extract Python code from Groq's response."""
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
# Install Anthropic's Python client first
# pip install anthropic

import os
import logging
from typing import Dict, Any, Optional, List

# import anthropic
from anthropic.types import ContentBlockParam
import groq

from agents.base_agent import BaseAgent
from utils.prompt_templates import OPENAI_PROMPT_TEMPLATES

logger = logging.getLogger(__name__)

class GroqAgent(BaseAgent):
    """
    Agent implementation using Anthropic's Claude 3.5
    """
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """Initialize the Open AI agent."""
        super().__init__(config_path)
        self.client = None
        self.model_name = self.config.get("models", {}).get("openai", {}).get("model_name", "gpt-4.1")
        self.max_tokens = self.config.get("models", {}).get("openai", {}).get("max_tokens", 1024)
        self.temperature = self.config.get("models", {}).get("openai", {}).get("temperature", 0)
        self.api_key = os.getenv("GROQ_API_KEY", "")
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
            test_response = self.client.messages.create(
                model=self.model_name,
                max_tokens=10,
                messages=[{"role": "user", "content": "Hello"}]
            )
            logger.info(f"Claude agent initialized successfully with model: {self.model_name}")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize Claude agent: {e}")
            return False
            
    def generate_code(self, prompt: str, dataset_info: Dict[str, Any]) -> str:
        """Generate Python code using Open AI."""
        if not self.client:
            if not self.initialize():
                return "Error: Claude agent is not initialized. Please check your API key."
        
        formatted_dataset_info = self.format_dataset_info(dataset_info)
        full_prompt = self.prompt_templates["code_generation"].format(
            user_prompt=prompt,
            dataset_info=formatted_dataset_info
        )
        
        try:
            response = self.client.messages.create(
                model=self.model_name,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                messages=[{"role": "user", "content": full_prompt}],
                system="You are a senior data scientist specialized in Python programming for data analysis. Your task is to generate high-quality, efficient, and well-documented Python code to address user questions about their dataset. Focus on creating code that is robust, handles errors gracefully, and produces insightful results. Always include explanatory comments. Only include code in your response, no additional explanations."
            )
            
            # Extract code from the response
            code = self._extract_code_from_response(response.content)
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
            response = self.client.messages.create(
                model=self.model_name,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                messages=[{"role": "user", "content": full_prompt}],
                system="You are a helpful data science assistant that provides clear and accurate answers to questions about code, data analysis, and statistics. Provide explanations that are accessible but technically precise."
            )
            
            return response.content[0].text
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
            response = self.client.messages.create(
                model=self.model_name,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                messages=[{"role": "user", "content": full_prompt}],
                system="You are a Python code optimization expert. Your task is to improve existing Python code based on user feedback while maintaining its core functionality. Focus on clarity, efficiency, and best practices. Only include the improved code in your response, no additional explanations."
            )
            
            # Extract code from the response
            improved_code = self._extract_code_from_response(response.content)
            return improved_code
        except Exception as e:
            logger.error(f"Error improving code with Claude: {e}")
            return f"Error improving code: {str(e)}"
    
    def _extract_code_from_response(self, content: List[ContentBlockParam]) -> str:
        """Extract Python code from Claude's response."""
        code = ""
        for block in content:
            if block.type == "text":
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
                if block.type == "text" and ("import " in block.text or "def " in block.text or "class " in block.text):
                    code = block.text
                    
        return code.strip()