# """
# OpenAI Agent - Implementation for GPT-4 models
# """

# import os
# import logging
# from typing import Dict, Any, Optional
# import json

# from openai import OpenAI
# from agents.base_agent import BaseAgent

# logger = logging.getLogger(__name__)

# class OpenAIAgent(BaseAgent):
#     """
#     Agent implementation using OpenAI's GPT-4 models.
#     """
    
#     def __init__(self, config_path: str = "config/config.yaml"):
#         """Initialize the OpenAI agent."""
#         super().__init__(config_path)
#         self.client = None
#         self.model_name = self.config.get("models", {}).get("openai", {}).get("model_name", "gpt-4.1")
        
#         # Fix the max_tokens parsing issue
#         max_tokens_value = self.config.get("models", {}).get("openai", {}).get("max_tokens", 1024)
#         if isinstance(max_tokens_value, str):
#             # Handle template format ${VAR:default}
#             if max_tokens_value.startswith("${") and ":" in max_tokens_value:
#                 # Extract default value after colon
#                 default_value = max_tokens_value.split(":", 1)[1].rstrip("}")
#                 self.max_tokens = int(default_value)
#             else:
#                 # Try simple conversion or use default
#                 try:
#                     self.max_tokens = int(max_tokens_value)
#                 except ValueError:
#                     self.max_tokens = 4096
#         else:
#             self.max_tokens = int(max_tokens_value)
        
#         # Ensure temperature is floating point
#         temp_value = self.config.get("models", {}).get("openai", {}).get("temperature", 0.2)
#         self.temperature = float(temp_value) if isinstance(temp_value, (int, float, str)) else 0.2
        
#         self.api_key = os.getenv("OPENAI_API_KEY", "")
#         self.last_prompt_tokens = 0
#         self.last_completion_tokens = 0
        
#         # Define prompt templates
#         self.prompt_templates = {
#         "code_generation": (
#             "Generate Python code for the following data analysis task:\n\n"
#             "User query: {user_prompt}\n\n"
#             "Dataset information:\n{dataset_info}\n\n"
#             "Requirements:\n"
#             "1. The dataset is available at '/sandbox/data.csv'. Always use this path to load the data.\n"
#             "2. Use pandas, numpy, matplotlib, seaborn, and scikit-learn as needed\n"
#             "3. Include visualizations when appropriate\n"
#             "4. Handle potential errors and edge cases\n"
#             "5. Include clear comments explaining your approach\n"
#             "6. Ensure the code is efficient and follows best practices\n"
#             "7. DO NOT use libraries that aren't pre-installed (like pandas_profiling)\n\n"
#             "Please respond with well-documented Python code that addresses the query."
#         ),
#         # self.prompt_templates = {
#         #     "code_generation": (
#         #         "Generate Python code for the following data analysis task:\n\n"
#         #         "User query: {user_prompt}\n\n"
#         #         "Dataset information:\n{dataset_info}\n\n"
#         #         "Requirements:\n"
#         #         "1. The dataset is already loaded as a pandas DataFrame named 'df'\n"
#         #         "2. Use pandas, numpy, matplotlib, seaborn, and scikit-learn as needed\n"
#         #         "3. Include visualizations when appropriate\n"
#         #         "4. Handle potential errors and edge cases\n"
#         #         "5. Include clear comments explaining your approach\n"
#         #         "6. Ensure the code is efficient and follows best practices\n\n"
#         #         "Please respond with well-documented Python code that addresses the query."
#         #     ),
#             "question_answering": (
#                 "Please answer the following question about data analysis or Python code:\n\n"
#                 "Question: {user_question}\n\n"
#                 "Context (if relevant):\n{context}\n\n"
#                 "Provide a clear, accurate, and detailed answer."
#             ),
#             "code_improvement": (
#                 "I have some Python code for data analysis that needs improvement:\n\n"
#                 "```python\n{original_code}\n```\n\n"
#                 "User feedback: {user_feedback}\n\n"
#                 "Please provide an improved version of this code that addresses the feedback."
#                 "Make the code more robust, efficient, and well-documented where needed."
#             )
#         }
        
#     def initialize(self) -> bool:
#         """Initialize the OpenAI client."""
#         try:
#             if not self.api_key:
#                 logger.warning("OpenAI API key not found")
#                 return False
                
#             self.client = OpenAI(api_key=self.api_key)
#             logger.info(f"OpenAI agent initialized successfully with model: {self.model_name}")
#             return True
#         except Exception as e:
#             logger.error(f"Failed to initialize OpenAI agent: {e}")
#             return False
            
#     # def generate_code(self, prompt: str, dataset_info: Dict[str, Any]) -> str:
#     #     """Generate Python code using OpenAI."""
#     #     if not self.client:
#     #         if not self.initialize():
#     #             return "Error: OpenAI agent is not initialized. Please check your API key."
        
#     #     formatted_dataset_info = self.format_dataset_info(dataset_info)
#     #     full_prompt = self.prompt_templates["code_generation"].format(
#     #         user_prompt=prompt,
#     #         dataset_info=formatted_dataset_info
#     #     )
        
#     #     # System prompt
#     #     system_content = "You are an expert data scientist. Generate clear, efficient, and well-documented Python code for data analysis tasks."
        
#     #     # Check prompt length and adjust max_tokens to stay within model context limits
#     #     try:
#     #         import tiktoken
#     #         encoding = tiktoken.encoding_for_model(self.model_name)
#     #         system_tokens = len(encoding.encode(system_content))
#     #         prompt_tokens = len(encoding.encode(full_prompt))
#     #         total_input_tokens = system_tokens + prompt_tokens
            
#     #         # GPT-4 models typically have 8192 token limit
#     #         model_context_limit = 8192
#     #         # Reserve a buffer of 200 tokens for safety
#     #         buffer = 200
            
#     #         # Calculate available tokens for completion
#     #         available_tokens = max(model_context_limit - total_input_tokens - buffer, 1024)
#     #         # Cap at our configured max_tokens
#     #         completion_tokens = min(available_tokens, self.max_tokens)
            
#     #         logger.info(f"Using {completion_tokens} tokens for completion (prompt: {total_input_tokens} tokens)")
#     #     except ImportError:
#     #         # If tiktoken isn't available, use a conservative value
#     #         completion_tokens = min(4096, self.max_tokens)
#     #         logger.info(f"Tiktoken not available, using conservative completion limit: {completion_tokens}")
        
#     #     try:
#     #         response = self.client.chat.completions.create(
#     #             model=self.model_name,
#     #             messages=[
#     #                 {
#     #                     "role": "system", 
#     #                     "content": system_content
#     #                 },
#     #                 {
#     #                     "role": "user", 
#     #                     "content": full_prompt
#     #                 }
#     #             ],
#     #             max_tokens=completion_tokens,
#     #             temperature=self.temperature
#     #         )
            
#     #         # Store token usage
#     #         self.last_prompt_tokens = response.usage.prompt_tokens
#     #         self.last_completion_tokens = response.usage.completion_tokens
            
#     #         # Extract code from the response
#     #         code = self._extract_code_from_response(response.choices[0].message.content)
            
#     #         # Format output with prefix and code
#     #         prefix = "Here's an analysis approach for your data:"
#     #         if "```" not in response.choices[0].message.content:
#     #             # No code blocks, return the whole response
#     #             return {"prefix": prefix, "code": code}
            
#     #         # Process any text before the first code block as the prefix
#     #         content = response.choices[0].message.content
#     #         if "```python" in content:
#     #             parts = content.split("```python", 1)
#     #             if parts[0].strip():
#     #                 prefix = parts[0].strip()
#     #         elif "```" in content:
#     #             parts = content.split("```", 1)
#     #             if parts[0].strip():
#     #                 prefix = parts[0].strip()
                    
#     #         return {"prefix": prefix, "code": code}
            
#     #     except Exception as e:
#     #         logger.error(f"Error generating code with OpenAI: {e}")
#     #         return f"Error generating code: {str(e)}"

#     def generate_code(self, prompt: str, dataset_info: Dict[str, Any]) -> str:
#         """Generate Python code using OpenAI."""
#         if not self.client:
#             if not self.initialize():
#                 return "Error: OpenAI agent is not initialized. Please check your API key."
        
#         formatted_dataset_info = self.format_dataset_info(dataset_info)
#         full_prompt = self.prompt_templates["code_generation"].format(
#             user_prompt=prompt,
#             dataset_info=formatted_dataset_info
#         )
        
#         # System prompt
#         system_content = "You are an expert data scientist. Generate clear, efficient, and well-documented Python code for data analysis tasks."
        
#         # Check prompt length and adjust max_tokens to stay within model context limits
#         try:
#             import tiktoken
#             encoding = tiktoken.encoding_for_model(self.model_name)
#             system_tokens = len(encoding.encode(system_content))
#             prompt_tokens = len(encoding.encode(full_prompt))
#             total_input_tokens = system_tokens + prompt_tokens
            
#             # GPT-4 models typically have 8192 token limit
#             model_context_limit = 8192
#             # Reserve a buffer of 200 tokens for safety
#             buffer = 200
            
#             # Calculate available tokens for completion
#             available_tokens = max(model_context_limit - total_input_tokens - buffer, 1024)
#             # Cap at our configured max_tokens
#             completion_tokens = min(available_tokens, self.max_tokens)
            
#             logger.info(f"Using {completion_tokens} tokens for completion (prompt: {total_input_tokens} tokens)")
#         except ImportError:
#             # If tiktoken isn't available, use a conservative value
#             completion_tokens = min(4096, self.max_tokens)
#             logger.info(f"Tiktoken not available, using conservative completion limit: {completion_tokens}")
        
#         try:
#             response = self.client.chat.completions.create(
#                 model=self.model_name,
#                 messages=[
#                     {
#                         "role": "system", 
#                         "content": system_content
#                     },
#                     {
#                         "role": "user", 
#                         "content": full_prompt
#                     }
#                 ],
#                 max_tokens=completion_tokens,
#                 temperature=self.temperature
#             )
            
#             # Store token usage
#             self.last_prompt_tokens = response.usage.prompt_tokens
#             self.last_completion_tokens = response.usage.completion_tokens
            
#             # Extract and return code from the response (fixed to return string, not dict)
#             code = self._extract_code_from_response(response.choices[0].message.content)
#             return code
            
#         except Exception as e:
#             logger.error(f"Error generating code with OpenAI: {e}")
#             return f"Error generating code: {str(e)}"
    

#     def answer_question(self, question: str, context: Optional[str] = None) -> str:
#         """Answer a question about code or dataset."""
#         if not self.client:
#             if not self.initialize():
#                 return "Error: OpenAI agent is not initialized. Please check your API key."
        
#         full_prompt = self.prompt_templates["question_answering"].format(
#             user_question=question,
#             context=context or ""
#         )
        
#         # System prompt
#         system_content = "You are a helpful data science expert providing clear and accurate answers."
        
#         # Calculate available tokens for completion
#         try:
#             import tiktoken
#             encoding = tiktoken.encoding_for_model(self.model_name)
#             system_tokens = len(encoding.encode(system_content))
#             prompt_tokens = len(encoding.encode(full_prompt))
#             total_input_tokens = system_tokens + prompt_tokens
            
#             model_context_limit = 8192
#             buffer = 200
            
#             available_tokens = max(model_context_limit - total_input_tokens - buffer, 1024)
#             completion_tokens = min(available_tokens, self.max_tokens)
            
#             logger.info(f"Q&A: Using {completion_tokens} tokens for completion (prompt: {total_input_tokens} tokens)")
#         except ImportError:
#             completion_tokens = min(4096, self.max_tokens)
#             logger.info(f"Q&A: Tiktoken not available, using conservative completion limit: {completion_tokens}")
        
#         try:
#             response = self.client.chat.completions.create(
#                 model=self.model_name,
#                 messages=[
#                     {
#                         "role": "system", 
#                         "content": system_content
#                     },
#                     {
#                         "role": "user", 
#                         "content": full_prompt
#                     }
#                 ],
#                 max_tokens=completion_tokens,
#                 temperature=self.temperature
#             )
            
#             # Store token usage
#             self.last_prompt_tokens = response.usage.prompt_tokens
#             self.last_completion_tokens = response.usage.completion_tokens
            
#             return response.choices[0].message.content
            
#         except Exception as e:
#             logger.error(f"Error answering question with OpenAI: {e}")
#             return f"Error answering question: {str(e)}"

#     def improve_code(self, code: str, feedback: str) -> str:
#         """Improve code based on feedback."""
#         if not self.client:
#             if not self.initialize():
#                 return "Error: OpenAI agent is not initialized. Please check your API key."
        
#         full_prompt = self.prompt_templates["code_improvement"].format(
#             original_code=code,
#             user_feedback=feedback
#         )
        
#         # System prompt
#         system_content = "You are an expert Python programmer specializing in improving data science code."
        
#         # Calculate available tokens for completion
#         try:
#             import tiktoken
#             encoding = tiktoken.encoding_for_model(self.model_name)
#             system_tokens = len(encoding.encode(system_content))
#             prompt_tokens = len(encoding.encode(full_prompt))
#             total_input_tokens = system_tokens + prompt_tokens
            
#             model_context_limit = 8192
#             buffer = 200
            
#             available_tokens = max(model_context_limit - total_input_tokens - buffer, 1024)
#             completion_tokens = min(available_tokens, self.max_tokens)
            
#             logger.info(f"Improve: Using {completion_tokens} tokens for completion (prompt: {total_input_tokens} tokens)")
#         except ImportError:
#             completion_tokens = min(4096, self.max_tokens)
#             logger.info(f"Improve: Tiktoken not available, using conservative completion limit: {completion_tokens}")
        
#         try:
#             response = self.client.chat.completions.create(
#                 model=self.model_name,
#                 messages=[
#                     {
#                         "role": "system", 
#                         "content": system_content
#                     },
#                     {
#                         "role": "user", 
#                         "content": full_prompt
#                     }
#                 ],
#                 max_tokens=completion_tokens,
#                 temperature=self.temperature
#             )
            
#             # Store token usage
#             self.last_prompt_tokens = response.usage.prompt_tokens
#             self.last_completion_tokens = response.usage.completion_tokens
            
#             return self._extract_code_from_response(response.choices[0].message.content)
            
#         except Exception as e:
#             logger.error(f"Error improving code with OpenAI: {e}")
#             return f"Error improving code: {str(e)}"
        
#     def _extract_code_from_response(self, response_content: str) -> str:
#         """Extract Python code from OpenAI's response."""
#         code = ""
        
#         # Try to extract code blocks with ```python ... ``` format
#         if "```python" in response_content:
#             code_blocks = response_content.split("```python")
#             for block in code_blocks[1:]:  # Skip the first part before ```python
#                 if "```" in block:
#                     code += block.split("```")[0].strip() + "\n\n"
#         # If no python blocks found, look for any code blocks
#         elif "```" in response_content and not code:
#             code_blocks = response_content.split("```")
#             # Take the content between ``` markers (odd indices)
#             for i in range(1, len(code_blocks), 2):
#                 if i < len(code_blocks):
#                     code += code_blocks[i].strip() + "\n\n"
        
#         # If no code blocks found, check if the text appears to be code itself
#         if not code and ("import " in response_content or "def " in response_content or "class " in response_content):
#             code = response_content
                    
#         return code.strip()

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
    
    def __init__(self, config_path: str = "config/config.yaml", model_name: Optional[str] = None):
        """Initialize the Open AI agent."""
        super().__init__(config_path)
        self.client = None
        self.model_name = model_name or self.config.get("models", {}).get("openai", {}).get("model_name", "gpt-4.1")
        if self.model_name == "o4-mini":
            self.max_completion_tokens = self.config.get("models", {}).get("openai", {}).get("max_completion_tokens", 2048)
        else:
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
            if self.model_name == "o4-mini":
                test_response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[{"role": "user", "content": "Hello"}],
                    max_completion_tokens=10
                )
            else:
                test_response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[{"role": "user", "content": "Hello"}],
                    max_tokens=10
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
            if self.model_name == "o4-mini":
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    max_completion_tokens=self.max_completion_tokens,
                    messages=[
                        {"role": "system", "content" : "You are a senior data scientist specialized in Python programming for data analysis. Your task is to generate high-quality, efficient, and well-documented Python code to address user questions about their dataset. Focus on creating code that is robust, handles errors gracefully, and produces insightful results. Always include explanatory comments. Only include code in your response, no additional explanations."},
                        {"role": "user", "content": full_prompt}
                        ]
                )
            else:
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
            if self.model_name == "o4-mini":
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    max_completion_tokens=self.max_completion_tokens,
                    messages=[
                        {"role": "system", "content" : "You are a senior data scientist specialized in Python programming for data analysis. Your task is to generate high-quality, efficient, and well-documented Python code to address user questions about their dataset. Focus on creating code that is robust, handles errors gracefully, and produces insightful results. Always include explanatory comments. Only include code in your response, no additional explanations."},
                        {"role": "user", "content": full_prompt}
                        ]
                )
            else:
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
            if self.model_name == "o4-mini":
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    max_completion_tokens=self.max_completion_tokens,
                    messages=[
                        {"role": "system", "content" : "You are a senior data scientist specialized in Python programming for data analysis. Your task is to generate high-quality, efficient, and well-documented Python code to address user questions about their dataset. Focus on creating code that is robust, handles errors gracefully, and produces insightful results. Always include explanatory comments. Only include code in your response, no additional explanations."},
                        {"role": "user", "content": full_prompt}
                        ]
                )
            else:
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