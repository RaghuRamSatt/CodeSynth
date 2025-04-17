import os
import uuid
from typing import List, Dict
from typing_extensions import TypedDict
from dotenv import load_dotenv

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_groq.chat_models import ChatGroq
from langgraph.checkpoint.memory import MemorySaver
from llm_sandbox import SandboxSession
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()
api_key = os.getenv("GROQ_API_KEY")

# Define graph state
class GraphState(TypedDict):
    error: str
    messages: List
    generation: Dict
    iterations: int
    dataset_path: str

# Data model for structured output
class CodeOutput(BaseModel):
    prefix: str = Field(description="Description of approach")
    code: str = Field(description="Executable Python code")

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
    if state["error"] == "yes":
        state["messages"].append(("user", "Please retry, ensuring you use the CODE tool with a prefix and code."))
    solution = code_gen_chain.invoke({"messages": state["messages"]})
    print(solution['parsed'], 'solution object')
    state["messages"].append(("assistant", f"{solution['parsed'].prefix}\nCode: {solution['parsed'].code}"))
    state["generation"] = solution['parsed']
    state["iterations"] += 1
    return state


def code_check(state: GraphState) -> GraphState:
    code_text = state["generation"].code
    ds_path = state["dataset_path"]
    print('ds_path is - ', ds_path)
    with SandboxSession(lang="python", keep_template=True) as session:
        # copy dataset into sandbox
        session.copy_to_runtime(ds_path, "/sandbox/data.csv")
        try:
            output = session.run(code_text, libraries=["pandas", "numpy", "matplotlib", "seaborn", "scikit-learn"])
            state["error"] = "no"
            print(output.text, 'output text')
        except Exception as e:
            state["messages"].append(("assistant", f"Execution failed: {e}"))
            state["error"] = "yes"
    return state


def reflect(state: GraphState) -> GraphState:
    # optional reflection
    solution = code_gen_chain.invoke({"messages": state["messages"]})
    state["messages"].append(("assistant", f"Reflection: {solution}"))
    return state


def decide_finish(state: GraphState) -> str:
    return 'end' if state["error"] == "no" or state["iterations"] >= 3 else "generate"

# Build graph
workflow = StateGraph(GraphState)
llm = ChatGroq(temperature=0.1, api_key=api_key, model="gemma2-9b-it")
structured_llm = llm.with_structured_output(CodeOutput, include_raw=True)
code_gen_chain = code_gen_prompt | structured_llm

workflow.add_node("generate", generate)
workflow.add_node("check_code", code_check)
workflow.add_node("reflect", reflect)
workflow.add_edge(START, "generate")
workflow.add_edge("generate", "check_code")
workflow.add_conditional_edges("check_code", decide_finish, {"end": END, "generate": "generate"})
workflow.add_edge("reflect", "generate")

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
    prompt = (
        f"Generate Python code for the following data analysis task:\n\n"
        f"User query: {user_query}\n\n"
        f"Dataset information:\n{info}\n\n"
        "The dataset is available at '/sandbox/data.csv'. "
        "Ensure your code reads it, e.g.: df = pd.read_csv('/sandbox/data.csv')\n"
        "IMPORTANT: Provide well-commented code, use pandas, numpy, matplotlib, seaborn, scikit-learn, "
        "and use the CODE tool with a prefix and code block only."
    )

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
