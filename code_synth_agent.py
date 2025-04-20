import os
import uuid
from typing import List, Dict
from typing_extensions import TypedDict
from dotenv import load_dotenv

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_groq.chat_models import ChatGroq
from langchain_anthropic.chat_models import ChatAnthropic
from langgraph.checkpoint.memory import MemorySaver
from llm_sandbox import SandboxSession
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()
# api_key = os.getenv("GROQ_API_KEY")
api_key = os.getenv("ANTHROPIC_API_KEY")

# Define graph state
class GraphState(TypedDict):
    error: str
    messages: List
    generation: Dict
    iterations: int
    dataset_path: str

# Data model for structured output
class CodeOutput(BaseModel):
    prefix: str = Field(description="Description of approach to the problem.")
    code: str = Field(description="Executable Python code. Do NOT include ```python``` or any other language tags.")

# Prompt template
code_gen_prompt = ChatPromptTemplate.from_messages([
    ("system", """
You are a coding assistant specializing in data analysis in Python.
Make sure all necessary imports and variables are defined.
Format your response with:
1) A prefix describing the solution
2) The functioning code block
Do not include any test logic or explanations outside the code.
Here is the user question:
"""),
    ("placeholder", "{messages}"),
])

# Helper functions for LangGraph nodes

def generate(state: GraphState) -> GraphState:

    try:
        print(f"Generation - {state['iterations']}")
        solution = code_gen_chain.invoke({"messages": state["messages"]})
    except Exception as e:
        state["messages"].append(("user", "Parsing error! Please ensure you use the CODE tool."))
        state["error"] = "yes"
        state["iterations"] += 1
        print(f"Error during code generation: {e}")
        return state
    
    if solution == 'parsing_error':
        state["messages"].append(("user", "Parsing error! Please ensure you use the CODE tool."))
        state["error"] = "yes"
    elif solution == 'tool_not_invoked':
        state["messages"].append(("user", "Tool not invoked! Please ensure you use the CODE tool."))
        state["error"] = "yes"
    else:
        state["messages"].append(("assistant", f"{solution['parsed'].prefix}\nCode: {solution['parsed'].code}"))
        state["generation"] = solution['parsed']
        state["error"] = "no"
        
    state["iterations"] += 1
    
    return state


def code_check(state: GraphState) -> GraphState:
    code_text = state["generation"].code
    ds_path = state["dataset_path"]
    print('ds_path is - ', ds_path)
    print('code_text is - ', code_text)
    with SandboxSession(lang="python", keep_template=True) as session:
        # copy dataset into sandbox
        session.copy_to_runtime(ds_path, "/sandbox/data.csv")
        try:
            output = session.run(code_text, libraries=["pandas", "numpy", "matplotlib", "seaborn", "scikit-learn"])
            state["error"] = "no"
            print(output.text, 'output text')
        except Exception as e:
            print(f"Error during execution: {e}")
        if "Traceback (most recent call last):" in output.text or "Error:" in output.text:
            state["messages"].append(("user", f"The execution failed due to the following error, fix the code: {output.text}"))
            state["error"] = "yes"
    return state


def check_groq_output(tool_output):
    """Check for parse error or failure to call the tool"""

    # Error with parsing
    if tool_output["parsing_error"]:
        # Report back output and parsing errors
        print(f"Error parsing your output! Be sure to invoke the tool. Output: {raw_output}. \n Parse error: {error}")
        raw_output = str(tool_output["raw"].content)
        error = tool_output["parsing_error"]
        return "parsing_error"

    # Tool was not invoked
    elif not tool_output["parsed"]:
        print("Failed to invoke tool!")
        return "tool_not_invoked"

    return tool_output

def decide_finish(state: GraphState) -> str:
    if state['error'] == "no":
        return 'end'
    elif state['error'] == "yes" and state["iterations"] < 3:
        return 'generate'
    else:
        return 'end'

def check_parsing_error(state: GraphState) -> str:
    if state['error'] == "no":
        return 'check_code'
    elif state['error'] == "yes" and state["iterations"] < 3:
        return 'generate'
    else:
        return 'check_code'

# Build graph
workflow = StateGraph(GraphState)
# llm = ChatGroq(temperature=0.1, api_key=api_key, model="gemma2-9b-it")
llm = ChatAnthropic(temperature=0.1, model="claude-3-5-sonnet-20240620", api_key=api_key)
structured_llm = llm.with_structured_output(CodeOutput, include_raw=True)
code_gen_chain = code_gen_prompt | structured_llm | check_groq_output

workflow.add_node("generate", generate)
workflow.add_node("check_code", code_check)

workflow.add_edge(START, "generate")
workflow.add_conditional_edges("generate", check_parsing_error, {"check_code": "check_code", "generate": "generate"})
workflow.add_conditional_edges("check_code", decide_finish, {"end": END, "generate": "generate"})

thread_cfg = {"configurable": {"thread_id": uuid.uuid4()}}
checkpointer = MemorySaver()
graph = workflow.compile(checkpointer=checkpointer)

# Synthesize helper

def synthesize(user_query: str, dataset_info: Dict, dataset_path: str) -> Dict:
    # Build dataset info text
    info = f"Dataset name: {dataset_info['name']}\n"
    info += f"Shape: {dataset_info['shape'][0]} rows, {dataset_info['shape'][1]} columns\n"
    info += "Columns:\n"
    for col in dataset_info['columns']:
        info += f"- {col['name']} ({col['type']})\n"
    info += f"\nSample data:\n{dataset_info['sample']}"

    # Assemble prompt
    prompt = f"""
    Generate Python code for the following data analysis task:\n\n
    User query: {user_query}\n\n
    Dataset information:\n{info}\n\n
    The dataset is available at '/sandbox/data.csv'. 
    Ensure your code reads it, e.g.: df = pd.read_csv('/sandbox/data.csv')\n
    IMPORTANT: Provide well-commented code, use pandas, numpy, matplotlib, seaborn, scikit-learn, 
    and use the CODE tool with a prefix and code block only.
    """
    
    # Initialize graph state
    initial_state: GraphState = {
        "messages": [("user", prompt)],
        "iterations": 0,
        "error": "",
        "dataset_path": dataset_path,
        "generation": {}
    }
    
    # Invoke graph
    final_state = graph.invoke(initial_state, config=thread_cfg)
    sol = final_state['generation']
    return {"prefix": sol.prefix, "code": sol.code}

if __name__ == "__main__":
    # CLI fallback
    while True:
        ui = input("User: ")
        if ui.lower() in ['q', 'quit', 'exit']:
            break
        for ev in graph.stream({"messages": [("user", ui)], "iterations": 0, "error": "", "dataset_path": ""}, config=thread_cfg):
            for v in ev.values():
                print("Assistant:", v['messages'][-1])
