"""
Multi-Agent LangGraph Synthesis Module

This module combines the LangGraph workflow architecture from code_synth_agent.py
with the flexible multi-agent support from the agents/ directory.
"""

import os
import uuid
import logging
import traceback
from typing import List, Dict, Optional, Any, Type, Union
from typing_extensions import TypedDict

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from pydantic import BaseModel, Field

# Import agent implementations
from agents.base_agent import BaseAgent
from agents.claude_agent import ClaudeAgent
from agents.opensource_agent import OpenSourceAgent
from agents.groq_agent import GroqAgent
from agents.openai_agent import OpenAIAgent

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Define graph state
class GraphState(TypedDict):
    error: str
    messages: List
    generation: Dict
    iterations: int
    dataset_path: str
    agent_type: str

# Data model for structured output
class CodeOutput(BaseModel):
    prefix: str = Field(description="Description of approach")
    code: str = Field(description="Executable Python code")

# Agent factory to get the right implementation
# def get_agent(agent_type: str) -> BaseAgent:
#     """
#     Factory function to create an agent based on the specified type.
    
#     Args:
#         agent_type: Type of agent to create (claude, opensource, or openai)
        
#     Returns:
#         An initialized agent instance
#     """
#     if agent_type == "claude":
#         agent = ClaudeAgent()
#     elif agent_type == "opensource":
#         agent = OpenSourceAgent()
#     else:
#         # Default to Claude if type not recognized
#         logger.warning(f"Unknown agent type '{agent_type}', defaulting to Claude")
#         agent = ClaudeAgent()
    
#     # Initialize agent
#     success = agent.initialize()
#     if not success:
#         logger.error(f"Failed to initialize {agent_type} agent")
    
#     return agent

def get_agent(agent_type: str) -> BaseAgent:
    """
    Factory function to create an agent based on the specified type.
    
    Args:
        agent_type: Type of agent to create (claude, groq-llama, groq-mixtral, or opensource)
        
    Returns:
        An initialized agent instance
    """
    if agent_type == "claude":
        agent = ClaudeAgent()
    elif agent_type.startswith("groq"):
        from agents.groq_agent import GroqAgent
        agent = GroqAgent()
        # Set the model based on selection
        if agent_type == "groq-llama":
            agent.model_name = "llama-3.3-70b-versatile"
        elif agent_type == "groq-gemma":
            agent.model_name = "gemma2-9b-it" 
    elif agent_type.startswith("openai"):
        # Set the model based on selection
        if agent_type == "openai-gpt4.1":
            model_name = "gpt-4.1"
        elif agent_type == "openai-o4mini":
            model_name = "o4-mini" 
        agent = OpenAIAgent(model_name=model_name)
    elif agent_type == "opensource":
        agent = OpenSourceAgent()
    else:
        # Default to Claude if type not recognized
        logger.warning(f"Unknown agent type '{agent_type}', defaulting to Claude")
        agent = ClaudeAgent()
    
    # Initialize agent
    success = agent.initialize()
    if not success:
        logger.error(f"Failed to initialize {agent_type} agent")
    
    return agent


# Helper functions for LangGraph nodes

# def generate(state: GraphState) -> GraphState:
    # """
    # Generate code using the specified agent type.
    # """
    # agent = get_agent(state["agent_type"])
    
    # if state["error"] == "yes":
    #     # If there was an error, add feedback to the next generation
    #     state["messages"].append(("user", "Please retry with the corrections. Ensure your code handles edge cases and runs without errors."))
    
    # # Extract the last user message
    # user_message = ""
    # for msg in state["messages"]:
    #     if msg[0] == "user":
    #         user_message = msg[1]
    
    # # Get dataset info from the prompt
    # dataset_info = {}
    # if "Dataset information:" in user_message:
    #     parts = user_message.split("Dataset information:")
    #     if len(parts) > 1:
    #         dataset_info_text = parts[1].strip()
    #         # Parse basic dataset info
    #         dataset_info = {
    #             "name": "dataset",
    #             "sample": dataset_info_text,
    #             "columns": []
    #         }
    
    # # Generate code using the agent
    # try:
    #     generated_code = agent.generate_code(user_message, dataset_info)
        
    #     # Parse into prefix and code - Claude agent already does this extraction
    #     prefix = "Here's a solution for your data analysis task:"
    #     code = generated_code
        
    #     if isinstance(generated_code, dict) and "prefix" in generated_code and "code" in generated_code:
    #         prefix = generated_code["prefix"]
    #         code = generated_code["code"]
        
    #     state["messages"].append(("assistant", f"{prefix}\n\nCode: {code}"))
    #     state["generation"] = {"prefix": prefix, "code": code}
    #     state["iterations"] += 1
        
    # except Exception as e:
    #     logger.error(f"Error in code generation: {e}")
    #     logger.error(traceback.format_exc())
    #     state["messages"].append(("assistant", f"Error generating code: {e}"))
    #     state["error"] = "yes"
    
    # return state

def generate(state: GraphState) -> GraphState:
    """
    Generate code using the specified agent type.
    """
    agent = get_agent(state["agent_type"])
    
    # Extract the last user message
    user_message = ""
    for msg in state["messages"]:
        if msg[0] == "user":
            user_message = msg[1]
    
    # Get dataset info from the prompt
    dataset_info = {}
    if "Dataset information:" in user_message:
        parts = user_message.split("Dataset information:")
        if len(parts) > 1:
            dataset_info_text = parts[1].strip()
            # Parse basic dataset info
            dataset_info = {
                "name": "dataset",
                "sample": dataset_info_text,
                "columns": []
            }
    
    # Check if this is a retry after error
    is_retry = False
    for msg in state["messages"]:
        if msg[0] == "assistant" and "Reflection:" in msg[1]:
            is_retry = True
            break
    
    # Generate code using the agent
    try:
        if is_retry:
            # Extract the most recent reflection
            reflection = ""
            for msg in reversed(state["messages"]):
                if msg[0] == "assistant" and "Reflection:" in msg[1]:
                    reflection = msg[1].replace("Reflection: ", "")
                    break
            
            # Customize the prompt with reflection guidance
            error_context = f"Previous code had errors. {reflection}"
            user_message = f"{user_message}\n\n{error_context}" if user_message else error_context
        
        generated_code = agent.generate_code(user_message, dataset_info)
        
        # Parse into prefix and code
        prefix = "Here's a solution for your data analysis task:"
        code = generated_code
        
        if isinstance(generated_code, dict) and "prefix" in generated_code and "code" in generated_code:
            prefix = generated_code["prefix"]
            code = generated_code["code"]
        
        state["messages"].append(("assistant", f"{prefix}\n\nCode: {code}"))
        state["generation"] = {"prefix": prefix, "code": code}
        state["iterations"] += 1
        
    except Exception as e:
        logger.error(f"Error in code generation: {e}")
        logger.error(traceback.format_exc())
        state["messages"].append(("assistant", f"Error generating code: {e}"))
        state["error"] = "yes"
    
    return state

def code_check(state: GraphState) -> GraphState:
    """
    Check generated code by executing it in the sandbox.
    """
    from llm_sandbox import SandboxSession
    
    code_text = state["generation"].get("code", "")
    ds_path = state["dataset_path"]
    
    if not code_text or not ds_path:
        state["error"] = "yes"
        state["messages"].append(("assistant", "Error: No code or dataset provided"))
        return state
    
    logger.info(f"Testing code with dataset at: {ds_path}")
    
    with SandboxSession(lang="python", keep_template=True) as session:
        # Copy dataset into sandbox
        try:
            session.copy_to_runtime(ds_path, "/sandbox/data.csv")
            
            # Execute code with necessary libraries
            output = session.run(
                code_text, 
                libraries=["pandas", "numpy", "matplotlib", "seaborn", "scikit-learn"]
            )
            
            state["error"] = "no"
            logger.info("Code execution successful")
            
        except Exception as e:
            error_msg = str(e)
            state["messages"].append(("assistant", f"Execution failed: {error_msg}"))
            state["error"] = "yes"
            logger.error(f"Code execution failed: {error_msg}")
    
    return state

# def reflect(state: GraphState) -> GraphState:
#     """
#     Optional reflection step to improve the code (not currently used).
#     """
#     # Not implemented yet
#     return state

def reflect(state: GraphState) -> GraphState:
    """
    Reflection step to analyze errors and improve the code.
    """
    agent = get_agent(state["agent_type"])
    
    # Extract the current code
    current_code = state["generation"].get("code", "")
    
    # Extract all error messages
    error_messages = []
    for msg in reversed(state["messages"]):
        if msg[0] == "assistant" and ("Error" in msg[1] or "failed" in msg[1].lower()):
            error_messages.append(msg[1])
            if len(error_messages) >= 2:  # Get the 2 most recent error messages
                break
    
    error_context = "\n".join(error_messages)
    
    # Create reflection prompt
    reflection_prompt = (
        f"The code generated has errors. Here are the error messages:\n\n{error_context}\n\n"
        f"Original code:\n```python\n{current_code}\n```\n\n"
        f"Please analyze what's wrong and suggest specific fixes. Focus on syntax errors, "
        f"type errors (like 'numpy.float64 object has no attribute index'), and logical errors. "
        f"Provide a detailed explanation of what's wrong and how to fix it."
    )
    
    try:
        # Generate reflection using the agent
        reflection = agent.answer_question(reflection_prompt)
        
        # Format the reflection for next generation
        formatted_reflection = (
            "Based on the previous error, here's what needs to be fixed:\n"
            f"{reflection}\n"
            "Please generate an improved version with these fixes."
        )
        
        # Add reflection to messages
        state["messages"].append(("assistant", f"Reflection: {formatted_reflection}"))
        state["messages"].append(("user", f"Fix the code based on this reflection. Complete code only."))
        
        logger.info("Reflection generated successfully")
        
    except Exception as e:
        logger.error(f"Error in reflection: {e}")
        state["messages"].append(("assistant", f"Error during reflection: {str(e)}"))
        # Add a simplified message to help next generation
        state["messages"].append(("user", "Please fix the code. Pay special attention to data types and method calls."))
    
    return state

def decide_finish(state: GraphState) -> str:
    """
    Decide whether to finish or retry code generation.
    """
    return 'end' if state["error"] == "no" or state["iterations"] >= 3 else "generate"

# Build the LangGraph workflow
# def build_workflow():
#     workflow = StateGraph(GraphState)
    
#     # Add nodes
#     workflow.add_node("generate", generate)
#     workflow.add_node("check_code", code_check)
#     workflow.add_node("reflect", reflect)
    
#     # Add edges
#     workflow.add_edge(START, "generate")
#     workflow.add_edge("generate", "check_code")
#     # workflow.add_conditional_edges("check_code", decide_finish, {"end": END, "generate": "generate"})
#     workflow.add_conditional_edges(
#     "check_code", 
#     decide_finish, 
#     {
#         "end": END,  # Success path
#         "generate": "reflect"  # Failure path now goes to reflect first
#     }
#     )
    
#     workflow.add_edge("reflect", "generate")
    
#     # Compile the graph
#     thread_cfg = {"configurable": {"thread_id": uuid.uuid4()}}
#     checkpointer = MemorySaver()
#     return workflow.compile(checkpointer=checkpointer), thread_cfg

def build_workflow():
    workflow = StateGraph(GraphState)
    
    # Add nodes
    workflow.add_node("generate", generate)
    workflow.add_node("check_code", code_check)
    workflow.add_node("reflect", reflect)
    
    # Add edges
    workflow.add_edge(START, "generate")
    workflow.add_edge("generate", "check_code")
    
    
    workflow.add_conditional_edges(
        "check_code", 
        decide_finish, 
        {
            "end": END,  # Success path
            "generate": "reflect"  # Failure path goes to reflect first
        }
    )
    
    # Connect reflection back to generation
    workflow.add_edge("reflect", "generate")
    
    # Compile the graph
    thread_cfg = {"configurable": {"thread_id": uuid.uuid4()}}
    checkpointer = MemorySaver()
    return workflow.compile(checkpointer=checkpointer), thread_cfg

# Pre-build the graph
graph, thread_cfg = build_workflow()

# Main synthesis function to be called by the app
def synthesize(
    user_query: str, 
    dataset_info: Dict, 
    dataset_path: str,
    agent_type: str = "claude"
) -> Dict:
    """
    Synthesize code for a data analysis task using the LangGraph workflow.
    
    Args:
        user_query: The user's natural language query
        dataset_info: Information about the dataset structure
        dataset_path: Path to the dataset file
        agent_type: Type of agent to use (claude, opensource)
        
    Returns:
        Dictionary containing the generated prefix and code
    """
    # Build dataset info text
    info = f"Dataset name: {dataset_info.get('name', 'Unknown')}\n"
    info += f"Shape: {dataset_info.get('shape', (0, 0))[0]} rows, {dataset_info.get('shape', (0, 0))[1]} columns\n"
    info += "Columns:\n"
    
    for col in dataset_info.get('columns', []):
        info += f"- {col.get('name', 'Unknown')} ({col.get('type', 'Unknown')})\n"
    
    info += f"\nSample data:\n{dataset_info.get('sample', '')}"

    # Assemble prompt
    prompt = (
        f"Generate Python code for the following data analysis task:\n\n"
        f"User query: {user_query}\n\n"
        f"Dataset information:\n{info}\n\n"
        "The dataset is available at '/sandbox/data.csv'. "
        "Ensure your code reads it, e.g.: df = pd.read_csv('/sandbox/data.csv')\n"
        "IMPORTANT: Provide well-commented code, use pandas, numpy, matplotlib, seaborn, scikit-learn, "
        "and ensure code is robust and handles edge cases."
    )

    # Initialize graph state
    initial_state: GraphState = {
        "messages": [("user", prompt)],
        "iterations": 0,
        "error": "",
        "dataset_path": dataset_path,
        "generation": {},
        "agent_type": agent_type
    }
    
    try:
        # Invoke graph
        final_state = graph.invoke(initial_state, config=thread_cfg)
        generation = final_state.get('generation', {})
        
        return {
            "prefix": generation.get('prefix', "Code analysis approach"),
            "code": generation.get('code', "# No code was generated")
        }
    except Exception as e:
        logger.error(f"Error in synthesis: {e}")
        logger.error(traceback.format_exc())
        return {
            "prefix": f"Error: {str(e)}",
            "code": "# An error occurred during code synthesis"
        }

if __name__ == "__main__":
    # CLI mode for testing
    import argparse
    
    parser = argparse.ArgumentParser(description="Test the multi-agent synthesis")
    parser.add_argument("--agent", default="claude", help="Agent type: claude or opensource")
    args = parser.parse_args()
    
    while True:
        query = input("Query (or 'exit' to quit): ")
        if query.lower() in ['exit', 'quit', 'q']:
            break
            
        # Test with a sample dataset
        import pandas as pd
        import os
        
        # Use iris dataset for testing
        from sklearn.datasets import load_iris
        iris = load_iris()
        df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
        df['species'] = pd.Categorical.from_codes(iris.target, iris.target_names)
        
        # Save to temp file
        os.makedirs("data/temp", exist_ok=True)
        path = "data/temp/iris_test.csv"
        df.to_csv(path, index=False)
        
        # Create dataset info
        dataset_info = {
            "name": "iris",
            "shape": df.shape,
            "columns": [
                {"name": col, "type": str(df[col].dtype)} for col in df.columns
            ],
            "sample": df.head(5).to_string()
        }
        
        result = synthesize(query, dataset_info, path, args.agent)
        print("\nPrefix:")
        print(result["prefix"])
        print("\nCode:")
        print(result["code"])