"""
Base Agent Module - Provides the foundation for all LLM agents
"""

import os
import yaml
import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional

# Setup logging
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

class BaseAgent(ABC):
    """
    Abstract base class for LLM agents in the system.
    All model-specific agents should inherit from this class.
    """
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """
        Initialize the base agent with configuration settings.
        
        Args:
            config_path: Path to the configuration file
        """
        self.config = self._load_config(config_path)
        self.logger = logger
        self.context = {}  # Store conversation context
        
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """
        Load configuration from YAML file.
        
        Args:
            config_path: Path to the configuration file
            
        Returns:
            Dictionary containing configuration settings
        """
        try:
            with open(config_path, 'r') as file:
                config = yaml.safe_load(file)
                
            # Process environment variables in config
            for key, value in os.environ.items():
                if key in config:
                    config[key] = value
                    
            return config
        except Exception as e:
            logger.error(f"Error loading configuration: {e}")
            # Return default configuration as fallback
            return {
                "app": {"name": "Data Analysis Agent", "debug": False},
                "models": {},
                "database": {},
                "execution_engine": {"timeout": 30}
            }
    
    def update_context(self, key: str, value: Any) -> None:
        """
        Update the conversation context with new information.
        
        Args:
            key: Context key
            value: Context value
        """
        self.context[key] = value
        
    def get_context(self, key: str, default: Any = None) -> Any:
        """
        Retrieve value from conversation context.
        
        Args:
            key: Context key
            default: Default value if key not found
            
        Returns:
            Value from context or default
        """
        return self.context.get(key, default)
    
    def clear_context(self) -> None:
        """Clear the entire conversation context."""
        self.context = {}
        
    @abstractmethod
    def initialize(self) -> bool:
        """
        Initialize the agent and establish any necessary connections.
        
        Returns:
            True if initialization was successful, False otherwise
        """
        pass
    
    @abstractmethod
    def generate_code(self, prompt: str, dataset_info: Dict[str, Any]) -> str:
        """
        Generate Python code based on user prompt and dataset information.
        
        Args:
            prompt: User prompt for code generation
            dataset_info: Information about the dataset structure
            
        Returns:
            Generated Python code as string
        """
        pass
    
    @abstractmethod
    def answer_question(self, question: str, context: Optional[str] = None) -> str:
        """
        Answer a question about code or dataset.
        
        Args:
            question: User question
            context: Additional context (optional)
            
        Returns:
            Agent's response to the question
        """
        pass
    
    @abstractmethod
    def improve_code(self, code: str, feedback: str) -> str:
        """
        Improve existing code based on user feedback.
        
        Args:
            code: Existing Python code
            feedback: User feedback for improvement
            
        Returns:
            Improved Python code
        """
        pass
    
    def format_dataset_info(self, dataset_info: Dict[str, Any]) -> str:
        """
        Format dataset information for the prompt.
        
        Args:
            dataset_info: Dictionary containing dataset metadata
            
        Returns:
            Formatted string with dataset information
        """
        result = f"Dataset Information:\n"
        result += f"Name: {dataset_info.get('name', 'Unnamed')}\n"
        result += f"Shape: {dataset_info.get('shape', 'Unknown')}\n"
        
        # Add column information
        columns = dataset_info.get('columns', [])
        if columns:
            result += "Columns:\n"
            for col in columns:
                col_name = col.get('name', 'Unknown')
                col_type = col.get('type', 'Unknown')
                col_desc = col.get('description', '')
                result += f"- {col_name} ({col_type}): {col_desc}\n"
                
        # Add sample data if available
        sample = dataset_info.get('sample', '')
        if sample:
            result += f"\nSample Data:\n{sample}\n"
            
        return result