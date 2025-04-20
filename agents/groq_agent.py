
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