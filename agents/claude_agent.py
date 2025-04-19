# Install Anthropic's Python client first
# pip install anthropic

import os
import logging
from typing import Dict, Any, Optional, List

import anthropic
from anthropic.types import ContentBlockParam

from agents.base_agent import BaseAgent
from utils.prompt_templates import CLAUDE_PROMPT_TEMPLATES

logger = logging.getLogger(__name__)

class ClaudeAgent(BaseAgent):
    """
    Agent implementation using Anthropic's Claude 3.5
    """
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """Initialize the Claude agent."""
        super().__init__(config_path)
        self.client = None
        self.model_name = self.config.get("models", {}).get("claude", {}).get("model_name", "claude-3-5-sonnet-20240620")
        self.max_tokens = self.config.get("models", {}).get("claude", {}).get("max_tokens", 4096)
        self.temperature = self.config.get("models", {}).get("claude", {}).get("temperature", 0.2)
        self.api_key = os.getenv("ANTHROPIC_API_KEY", "")
        self.prompt_templates = CLAUDE_PROMPT_TEMPLATES
        
    def initialize(self) -> bool:
        """Initialize the Claude client."""
        try:
            # Try loading from .env file directly
            from dotenv import load_dotenv
            load_dotenv()
            
            self.api_key = os.getenv("ANTHROPIC_API_KEY", "")
            
            # Add debug logging
            logger.info(f"API key found: {'Yes' if self.api_key else 'No'}")
            
            if not self.api_key:
                logger.error("ANTHROPIC_API_KEY environment variable not set")
                return False
                    
            self.client = anthropic.Anthropic(api_key=self.api_key)
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
        """Generate Python code using Claude."""
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
                system="You are a senior data scientist specialized in Python programming for data analysis. Your task is to generate high-quality, efficient, and well-documented Python code to address user questions about their dataset. Focus on creating code that is robust, handles errors gracefully, and produces insightful results. Always include explanatory comments. Only include code in your response, no additional explanations. Always use the variable 'dataset_path' to load data with pd.read_csv(dataset_path)."
            )
            
            self.last_prompt_tokens = response.usage.input_tokens
            self.last_completion_tokens = response.usage.output_tokens
            
            # Extract code from the response
            code = self._extract_code_from_response(response.content)
            return code
        except Exception as e:
            logger.error(f"Error generating code with Claude: {e}")
            return f"Error generating code: {str(e)}"
            
    def answer_question(self, question: str, context: Optional[str] = None) -> str:
        """Answer a question about code or dataset."""
        if not self.client:
            if not self.initialize():
                return "Error: Claude agent is not initialized. Please check your API key."
        
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
            logger.error(f"Error answering question with Claude: {e}")
            return f"Error answering question: {str(e)}"
            
    def improve_code(self, code: str, feedback: str) -> str:
        """Improve existing code based on user feedback."""
        if not self.client:
            if not self.initialize():
                return "Error: Claude agent is not initialized. Please check your API key."
        
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