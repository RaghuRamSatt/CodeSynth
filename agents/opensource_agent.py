"""
Open-Source Agent - Implementation for open-source models like Phi-3 Mini
"""

import os
import logging
import requests
from typing import Dict, Any, Optional, List, Union
import json

from langchain_community.llms import HuggingFacePipeline
from langchain.prompts import PromptTemplate
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

from agents.base_agent import BaseAgent
from utils.prompt_templates import OPENSOURCE_PROMPT_TEMPLATES

logger = logging.getLogger(__name__)

class OpenSourceAgent(BaseAgent):
    """
    Agent implementation using open-source models like Phi-3 Mini or Mistral Small.
    Can work with either local models or hosted API endpoints.
    """
    
    def __init__(self, config_path: str = "config/config.yaml", model_type: str = None):
        """
        Initialize the open-source model agent.
        
        Args:
            config_path: Path to the configuration file
            model_type: Override model type from config (phi3-mini-4k or phi3-mini-128k)
        """
        super().__init__(config_path)
        
        # Get config settings
        model_config = self.config.get("models", {}).get("opensource", {})
        self.model_type = model_type or model_config.get("model_type", "phi3-mini-4k")
        self.local_model_path = os.getenv("PHI3_MODEL_PATH", model_config.get("local_model_path", ""))
        self.hosted_endpoint = os.getenv("PHI3_ENDPOINT", model_config.get("hosted_endpoint", ""))
        self.max_tokens = model_config.get("max_tokens", 4096)
        self.temperature = model_config.get("temperature", 0.2)
        
        self.prompt_templates = OPENSOURCE_PROMPT_TEMPLATES
        self.model = None
        self.tokenizer = None
        self.pipe = None
        self.is_local = bool(self.local_model_path)
        
        # Set model ID based on model type
        if self.model_type == "phi3-mini-4k":
            self.model_id = "microsoft/Phi-3-mini-4k-instruct"
        elif self.model_type == "phi3-mini-128k":
            self.model_id = "microsoft/Phi-3-mini-128k-instruct"
        else:
            logger.warning(f"Unknown model type: {self.model_type}. Defaulting to phi3-mini-4k")
            self.model_id = "microsoft/Phi-3-mini-4k-instruct"
        
    def initialize(self) -> bool:
        """
        Initialize the open-source model with optimized memory usage.
        """
        try:
            if self.is_local:
                # If specified path doesn't exist, try to download from HF
                if not os.path.exists(self.local_model_path):
                    logger.info(f"Local model path not found. Attempting to download {self.model_id} from Hugging Face")
                    self.local_model_path = self.model_id
                
                logger.info(f"Loading model from {self.local_model_path}")
                
                # Load tokenizer
                self.tokenizer = AutoTokenizer.from_pretrained(self.local_model_path)
                
                # Load model with 4-bit quantization for better memory efficiency
                from transformers import BitsAndBytesConfig
                
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype="float16",
                    bnb_4bit_use_double_quant=True,
                )
                
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.local_model_path,
                    device_map="auto",
                    quantization_config=bnb_config,
                    attn_implementation="eager"
                )
                
                # Create text generation pipeline with reduced context
                self.pipe = pipeline(
                    "text-generation",
                    model=self.model,
                    tokenizer=self.tokenizer,
                    max_new_tokens=512,  # Reduced for memory efficiency
                    temperature=self.temperature,
                    do_sample=True
                )
                
                # Import this locally to avoid requiring the package if not needed
                from langchain_community.llms import HuggingFacePipeline
                self.langchain_model = HuggingFacePipeline(pipeline=self.pipe)
                
                logger.info(f"Open-source model {self.model_type} loaded successfully")
                return True
                
            elif self.hosted_endpoint:
                # Using a hosted API endpoint
                # Test connection with a simple query
                response = self._call_hosted_api("Hello")
                if response and isinstance(response, str):
                    logger.info(f"Connected to hosted {self.model_type} endpoint successfully")
                    return True
                else:
                    logger.error(f"Failed to connect to hosted endpoint: {response}")
                    return False
            else:
                logger.error("Neither local model path nor hosted endpoint provided")
                return False
                
        except Exception as e:
            logger.error(f"Failed to initialize open-source model: {e}")
            return False
            
    def generate_code(self, prompt: str, dataset_info: Dict[str, Any]) -> str:
        """
        Generate Python code using the open-source model.
        
        Args:
            prompt: User prompt for code generation
            dataset_info: Information about the dataset structure
            
        Returns:
            Generated Python code as string
        """
        if not self.is_local and not self.hosted_endpoint:
            if not self.initialize():
                return "Error: Open-source model is not initialized."
        
        formatted_dataset_info = self.format_dataset_info(dataset_info)
        full_prompt = self.prompt_templates["code_generation"].format(
            user_prompt=prompt,
            dataset_info=formatted_dataset_info
        )
        
        try:
            if self.is_local:
                # Use the local model
                prompt_template = PromptTemplate.from_template(
                    "<|user|>\n{prompt}\n<|end|>\n<|assistant|>"
                )
                
                formatted_prompt = prompt_template.format(prompt=full_prompt)
                response = self.langchain_model.invoke(formatted_prompt)
                
                # Extract code from the response
                code = self._extract_code_from_response(response)
                
            else:
                # Use the hosted API
                response = self._call_hosted_api(full_prompt)
                code = self._extract_code_from_response(response)
                
            return code
        except Exception as e:
            logger.error(f"Error generating code with open-source model: {e}")
            return f"Error generating code: {str(e)}"
            
    def answer_question(self, question: str, context: Optional[str] = None) -> str:
        """
        Answer a question about code or dataset.
        
        Args:
            question: User question
            context: Additional context (optional)
            
        Returns:
            Model's response to the question
        """
        if not self.is_local and not self.hosted_endpoint:
            if not self.initialize():
                return "Error: Open-source model is not initialized."
        
        full_prompt = self.prompt_templates["question_answering"].format(
            user_question=question,
            context=context or ""
        )
        
        try:
            if self.is_local:
                # Use the local model
                prompt_template = PromptTemplate.from_template(
                    "<|user|>\n{prompt}\n<|end|>\n<|assistant|>"
                )
                
                formatted_prompt = prompt_template.format(prompt=full_prompt)
                response = self.langchain_model.invoke(formatted_prompt)
                
            else:
                # Use the hosted API
                response = self._call_hosted_api(full_prompt)
                
            return response
        except Exception as e:
            logger.error(f"Error answering question with open-source model: {e}")
            return f"Error answering question: {str(e)}"
            
    def improve_code(self, code: str, feedback: str) -> str:
        """
        Improve existing code based on user feedback.
        
        Args:
            code: Existing Python code
            feedback: User feedback for improvement
            
        Returns:
            Improved Python code
        """
        if not self.is_local and not self.hosted_endpoint:
            if not self.initialize():
                return "Error: Open-source model is not initialized."
        
        full_prompt = self.prompt_templates["code_improvement"].format(
            original_code=code,
            user_feedback=feedback
        )
        
        try:
            if self.is_local:
                # Use the local model
                prompt_template = PromptTemplate.from_template(
                    "<|user|>\n{prompt}\n<|end|>\n<|assistant|>"
                )
                
                formatted_prompt = prompt_template.format(prompt=full_prompt)
                response = self.langchain_model.invoke(formatted_prompt)
                
                # Extract code from the response
                improved_code = self._extract_code_from_response(response)
                
            else:
                # Use the hosted API
                response = self._call_hosted_api(full_prompt)
                improved_code = self._extract_code_from_response(response)
                
            return improved_code
        except Exception as e:
            logger.error(f"Error improving code with open-source model: {e}")
            return f"Error improving code: {str(e)}"
    
    def _call_hosted_api(self, prompt: str) -> str:
        """
        Call the hosted API endpoint.
        
        Args:
            prompt: The prompt to send to the API
            
        Returns:
            API response as string
        """
        payload = {
            "prompt": prompt,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature
        }
        
        try:
            response = requests.post(
                self.hosted_endpoint,
                json=payload,
                headers={"Content-Type": "application/json"}
            )
            
            if response.status_code == 200:
                return response.json().get("text", "")
            else:
                logger.error(f"API error: {response.status_code} - {response.text}")
                return f"API error: {response.status_code}"
                
        except Exception as e:
            logger.error(f"Error calling hosted API: {e}")
            return f"Error: {str(e)}"
    
    def _extract_code_from_response(self, response: str) -> str:
        """
        Extract Python code from the model's response.
        
        Args:
            response: Model response
            
        Returns:
            Extracted Python code as string
        """
        # Handle different code block formats
        if "```python" in response:
            code_blocks = response.split("```python")
            code = ""
            for block in code_blocks[1:]:  # Skip the first part before ```python
                if "```" in block:
                    code += block.split("```")[0].strip() + "\n\n"
            return code.strip()
            
        elif "```" in response:
            code_blocks = response.split("```")
            # Take the content between ``` markers (odd indices)
            code = ""
            for i in range(1, len(code_blocks), 2):
                if i < len(code_blocks):
                    code += code_blocks[i].strip() + "\n\n"
            return code.strip()
        
        # If no code blocks found, check if the text appears to be code itself
        elif "import " in response or "def " in response or "class " in response:
            # This might be code without markdown formatting
            return response.strip()
            
        # No code found
        return response.strip()