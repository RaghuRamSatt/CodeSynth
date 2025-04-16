"""
Enhanced Streamlit app for data analysis with Claude 3.5 - Fixed version
"""

import os
import logging
import time
import base64
import pandas as pd
import numpy as np
import streamlit as st
import anthropic
from dotenv import load_dotenv
from io import StringIO, BytesIO
import matplotlib.pyplot as plt
import seaborn as sns

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Set page title
st.set_page_config(
    page_title="Data Analysis LLM Agent",
    page_icon="📊",
    layout="wide"
)

# Initialize session state
def init_session_state():
    """Initialize session state variables if they don't exist."""
    if 'conversation' not in st.session_state:
        st.session_state.conversation = []
    if 'dataset' not in st.session_state:
        st.session_state.dataset = None
    if 'dataset_info' not in st.session_state:
        st.session_state.dataset_info = {}
    if 'dataset_path' not in st.session_state:
        st.session_state.dataset_path = None
    if 'generated_code' not in st.session_state:
        st.session_state.generated_code = ""
    if 'code_execution_results' not in st.session_state:
        st.session_state.code_execution_results = None
    if 'token_count' not in st.session_state:
        st.session_state.token_count = {"input": 0, "output": 0}

init_session_state()

# App title and description
st.title("Data Analysis LLM Agent")
st.markdown("""
This application uses Claude 3.5 to generate Python code for data analysis based on your natural language queries. 
Simply load a dataset, ask questions about your data, and get Python code and visualizations.
""")

# Sidebar
with st.sidebar:
    st.header("Configuration")
    
    # API key status
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if api_key:
        st.success("Claude API Key: ✓ Connected")
    else:
        st.error("Claude API Key: ✗ Missing")
        st.info("Add ANTHROPIC_API_KEY to your .env file")
    
    # Dataset upload
    st.subheader("Upload Dataset")
    uploaded_file = st.file_uploader(
        "Choose a CSV, Excel, or JSON file", 
        type=["csv", "xlsx", "xls", "json"]
    )
    
    if uploaded_file is not None:
        try:
            # Save the uploaded file
            save_path = os.path.join("data/user_datasets", uploaded_file.name)
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            
            with open(save_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            # Load the dataset
            if uploaded_file.name.endswith('csv'):
                df = pd.read_csv(uploaded_file)
            elif uploaded_file.name.endswith(('xls', 'xlsx')):
                df = pd.read_excel(uploaded_file)
            elif uploaded_file.name.endswith('json'):
                df = pd.read_json(uploaded_file)
            else:
                df = None
                st.error("Unsupported file format")
            
            if df is not None:
                st.session_state.dataset = df
                st.session_state.dataset_path = save_path
                
                # Extract dataset information
                st.session_state.dataset_info = {
                    "name": uploaded_file.name,
                    "shape": df.shape,
                    "columns": [
                        {
                            "name": col,
                            "type": str(df[col].dtype),
                            "description": "",
                            "sample": str(df[col].iloc[0]) if len(df) > 0 else ""
                        }
                        for col in df.columns
                    ],
                    "sample": df.head(5).to_string()
                }
                
                st.success(f"Dataset loaded: {uploaded_file.name}")
                st.write(f"Shape: {df.shape[0]} rows × {df.shape[1]} columns")
                
        except Exception as e:
            st.error(f"Error loading dataset: {str(e)}")
    
    # Sample dataset selection
    st.subheader("Or Use Sample Dataset")
    sample_datasets = ["Iris", "Diamonds", "Tips", "Planets"]
    selected_sample = st.selectbox("Select a sample dataset:", ["None"] + sample_datasets)
    
    if selected_sample != "None":
        try:
            if selected_sample == "Iris":
                # Use sklearn's built-in iris dataset
                from sklearn.datasets import load_iris
                iris = load_iris()
                df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
                df['species'] = pd.Categorical.from_codes(iris.target, iris.target_names)
                filename = "iris.csv"
            elif selected_sample == "Diamonds":
                # Use seaborn's diamonds dataset
                df = sns.load_dataset('diamonds')
                filename = "diamonds.csv"
            elif selected_sample == "Tips":
                # Use seaborn's tips dataset
                df = sns.load_dataset('tips')
                filename = "tips.csv"
            elif selected_sample == "Planets":
                # Use seaborn's planets dataset
                df = sns.load_dataset('planets')
                filename = "planets.csv"
            
            # Save to file
            os.makedirs("data/sample_datasets", exist_ok=True)
            file_path = f"data/sample_datasets/{filename}"
            df.to_csv(file_path, index=False)
            
            st.session_state.dataset = df
            st.session_state.dataset_path = file_path
            
            # Extract dataset information
            st.session_state.dataset_info = {
                "name": selected_sample,
                "shape": df.shape,
                "columns": [
                    {
                        "name": col,
                        "type": str(df[col].dtype),
                        "description": "",
                        "sample": str(df[col].iloc[0]) if len(df) > 0 else ""
                    }
                    for col in df.columns
                ],
                "sample": df.head(5).to_string()
            }
            
            st.success(f"Sample dataset loaded: {selected_sample}")
            st.write(f"Shape: {df.shape[0]} rows × {df.shape[1]} columns")
                
        except Exception as e:
            st.error(f"Error loading sample dataset: {str(e)}")
            logger.error(f"Error loading sample dataset: {e}", exc_info=True)
    
    # Usage metrics
    st.subheader("API Usage")
    st.info(f"Input tokens: {st.session_state.token_count['input']}")
    st.info(f"Output tokens: {st.session_state.token_count['output']}")
    
    # Estimated cost (Claude 3.5 Sonnet pricing)
    input_cost = st.session_state.token_count['input'] * 3.00 / 1_000_000
    output_cost = st.session_state.token_count['output'] * 15.00 / 1_000_000
    total_cost = input_cost + output_cost
    
    st.info(f"Estimated cost: ${total_cost:.4f}")
    
    # Clear conversation button
    if st.button("Clear Conversation"):
        st.session_state.conversation = []
        st.session_state.generated_code = ""
        st.session_state.code_execution_results = None
        st.success("Conversation cleared!")

# Function to execute code
def execute_code():
    """Execute the generated code and capture results."""
    if not st.session_state.generated_code:
        return
    
    with st.spinner("Running code..."):
        # Initialize result dictionary
        result = {
            "success": False,
            "output": "",
            "error": "",
            "figures": [],
            "execution_time": 0
        }
        
        # Set up output capture
        output_buffer = StringIO()
        
        # Prepare execution environment with all necessary imports and objects
        exec_globals = {
            "__builtins__": __builtins__,
            "print": lambda *args, **kwargs: print(*args, file=output_buffer, **kwargs),
            "df": st.session_state.dataset,
            "pd": pd,
            "np": np,
            "plt": plt,
            "sns": sns,
            "BytesIO": BytesIO,
            "StringIO": StringIO,
            "base64": base64
        }
        
        # Import common libraries and setup figure tracking
        exec_code = """
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import base64
from io import BytesIO, StringIO
from sklearn import datasets, metrics, model_selection, preprocessing, decomposition, cluster
import warnings
warnings.filterwarnings('ignore')

# Try to set a safe style, but don't fail if it doesn't work
try:
    plt.style.use('default')
except Exception as e:
    print(f"Warning: Could not set matplotlib style: {e}")

# Store original plt.figure function
_original_figure = plt.figure

# List to store figure references
_figures = []

# Override plt.figure to track figures
def _custom_figure(*args, **kwargs):
    fig = _original_figure(*args, **kwargs)
    _figures.append(fig)
    return fig

plt.figure = _custom_figure

# Handle plt.subplots as well
_original_subplots = plt.subplots
def _custom_subplots(*args, **kwargs):
    fig, ax = _original_subplots(*args, **kwargs)
    _figures.append(fig)
    return fig, ax

plt.subplots = _custom_subplots

# Function to get base64 encoded figures
def _get_figures_as_base64():
    results = []
    for fig in _figures:
        buf = BytesIO()
        fig.savefig(buf, format='png', dpi=100)
        buf.seek(0)
        img_str = base64.b64encode(buf.read()).decode('utf-8')
        results.append(img_str)
    return results

# Also capture any current figure at the end
def _capture_current_figure():
    current_fig = plt.gcf()
    if current_fig not in _figures:
        _figures.append(current_fig)

"""
        
        # Add the user's code
        exec_code += "\n" + st.session_state.generated_code + "\n"
        
        # Add code to get figures at the end
        exec_code += """
# Capture current figure if it exists
try:
    _capture_current_figure()
except:
    pass

# Capture all figures that were created
_captured_figures = _get_figures_as_base64()
"""
        
        # Run the code
        start_time = time.time()
        
        try:
            # Execute the code
            exec(exec_code, exec_globals)
            
            # Mark as successful
            result["success"] = True
            
            # Get captured figures
            if "_captured_figures" in exec_globals:
                result["figures"] = exec_globals["_captured_figures"]
                
        except Exception as e:
            result["success"] = False
            result["error"] = f"{type(e).__name__}: {str(e)}"
            import traceback
            result["error"] += f"\n{traceback.format_exc()}"
            
        finally:
            # Record execution time
            result["execution_time"] = time.time() - start_time
            
            # Get output
            result["output"] = output_buffer.getvalue()
            
        # Store results
        st.session_state.code_execution_results = result

# Main area
main_col1, main_col2 = st.columns([3, 2])

with main_col1:
    # Dataset preview if available
    if st.session_state.dataset is not None:
        st.subheader("Dataset Preview")
        st.dataframe(st.session_state.dataset.head(), use_container_width=True)
        
        # User input
        st.subheader("Ask About Your Data")
        user_input = st.text_area(
            "Enter your query about the dataset:",
            placeholder="Example: Create a scatter plot of sepal length vs. sepal width colored by species.",
            height=100
        )
        
        # Query buttons
        col1, col2, col3, col4 = st.columns([1, 1, 1, 1])
        with col1:
            submit_button = st.button("Generate Code", type="primary")
        with col2:
            run_button = st.button("Run Code", disabled=not st.session_state.generated_code)
        with col3:
            save_button = st.button("Save Code", disabled=not st.session_state.generated_code)
        with col4:
            new_query_button = st.button("New Query")
        
        # Process user input when submit button is clicked
        if submit_button and user_input:
            try:
                with st.spinner("Generating code with Claude 3.5..."):
                    # Check for API key
                    api_key = os.getenv("ANTHROPIC_API_KEY")
                    if not api_key:
                        st.error("ANTHROPIC_API_KEY not found in environment variables.")
                        logger.error("ANTHROPIC_API_KEY not found in environment variables.")
                    else:
                        # Initialize Anthropic client
                        client = anthropic.Anthropic(api_key=api_key)
                        
                        # Format dataset info for prompt
                        dataset_info_text = f"""
                        Dataset name: {st.session_state.dataset_info['name']}
                        Shape: {st.session_state.dataset_info['shape'][0]} rows, {st.session_state.dataset_info['shape'][1]} columns
                        Columns: 
                        """
                        
                        for col in st.session_state.dataset_info['columns']:
                            dataset_info_text += f"- {col['name']} ({col['type']})\n"
                        
                        dataset_info_text += f"\nSample data:\n{st.session_state.dataset.head(5).to_string()}"
                        
                        # Create prompt for Claude with special instructions to avoid problematic code
                        prompt = f"""
                        Generate Python code for the following data analysis task:
                        
                        User query: {user_input}
                        
                        Dataset information:
                        {dataset_info_text}
                        
                        The dataset is already loaded into a pandas DataFrame called 'df'.
                        
                        IMPORTANT REQUIREMENTS:
                        1. Generate well-documented Python code with detailed comments
                        2. Use pandas, numpy, matplotlib, seaborn, and scikit-learn as needed
                        3. Include proper error handling where appropriate
                        4. Make sure to create informative visualizations with proper labels and titles
                        5. DO NOT use plt.style.use('seaborn') - use default styles or explicitly set colors/styles
                        6. When creating figures, always use plt.figure() or plt.subplots() to create a new figure
                        7. Only provide the code (no explanations before or after the code)
                        
                        The code will be executed with all necessary libraries already imported (pandas, numpy, matplotlib, seaborn, scikit-learn).
                        """
                        
                        # Add user message to conversation
                        st.session_state.conversation.append({
                            "role": "user", 
                            "content": user_input
                        })
                        
                        # Update token counts (estimated)
                        prompt_tokens = len(prompt) // 4  # Rough estimate
                        st.session_state.token_count["input"] += prompt_tokens
                        
                        # Send request to Claude
                        start_time = time.time()
                        response = client.messages.create(
                            model="claude-3-5-sonnet-20240620",
                            max_tokens=2000,
                            temperature=0.2,
                            messages=[{"role": "user", "content": prompt}]
                        )
                        
                        # Process the response
                        code_text = response.content[0].text
                        
                        # Extract code if it's wrapped in markdown
                        if "```python" in code_text:
                            code_parts = code_text.split("```python")
                            if len(code_parts) > 1:
                                code_text = code_parts[1].split("```")[0]
                        elif "```" in code_text:
                            code_parts = code_text.split("```")
                            if len(code_parts) > 1:
                                code_text = code_parts[1]
                        
                        # Store generated code
                        st.session_state.generated_code = code_text.strip()
                        
                        # Update token counts (estimated)
                        response_tokens = len(code_text) // 4  # Rough estimate
                        st.session_state.token_count["output"] += response_tokens
                        
                        # Add assistant message to conversation
                        st.session_state.conversation.append({
                            "role": "assistant",
                            "content": st.session_state.generated_code,
                            "type": "code"
                        })
                        
                        # Run the code automatically
                        execute_code()
            
            except Exception as e:
                st.error(f"Error generating code: {str(e)}")
                logger.error(f"Error generating code: {e}", exc_info=True)
        
        # Run code when run button is clicked
        if run_button and st.session_state.generated_code:
            execute_code()
        
        # Save code when save button is clicked
        if save_button and st.session_state.generated_code:
            # Create downloads directory if it doesn't exist
            os.makedirs("data/downloads", exist_ok=True)
            
            # Generate a filename based on the current time
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            filename = f"data_analysis_{timestamp}.py"
            filepath = os.path.join("data/downloads", filename)
            
            # Save the code to a file
            with open(filepath, "w") as f:
                f.write(st.session_state.generated_code)
            
            # Create a download button
            with open(filepath, "r") as f:
                code_content = f.read()
                
            st.download_button(
                label="Download Python File",
                data=code_content,
                file_name=filename,
                mime="text/plain"
            )
            
            st.success(f"Code saved as {filename}")
        
        # Clear current code for a new query
        if new_query_button:
            st.session_state.generated_code = ""
            st.session_state.code_execution_results = None
    else:
        st.info("Please load a dataset from the sidebar first.")

with main_col2:
    # Display generated code
    st.subheader("Generated Code")
    
    if st.session_state.generated_code:
        st.code(st.session_state.generated_code, language="python")
    else:
        st.info("Code will appear here after generation.")
    
    # Display execution results
    if hasattr(st.session_state, 'code_execution_results') and st.session_state.code_execution_results:
        st.subheader("Execution Results")
        
        results = st.session_state.code_execution_results
        
        # Display success/error status
        if results["success"]:
            st.success("Code executed successfully")
            st.write(f"Execution time: {results['execution_time']:.2f} seconds")
        else:
            st.error("Code execution failed")
            st.error(results["error"])
        
        # Display text output
        if results["output"]:
            st.subheader("Output")
            st.text(results["output"])
        
        # Display figures
        if results["figures"]:
            st.subheader("Visualizations")
            for i, fig_base64 in enumerate(results["figures"]):
                st.image(
                    BytesIO(base64.b64decode(fig_base64)),
                    caption=f"Figure {i+1}",
                    use_container_width=True
                )

# Display conversation history
st.subheader("Conversation History")
for message in st.session_state.conversation:
    if message["role"] == "user":
        st.markdown(f"**You:** {message['content']}")
    else:  # assistant
        if message.get("type") == "code":
            st.markdown("**Assistant:** Generated the following code:")
            with st.expander("Show code"):
                st.code(message["content"], language="python")
        else:
            st.markdown(f"**Assistant:** {message['content']}")

# Footer
st.markdown("---")
st.markdown(
    "Data Analysis LLM Agent - Using Claude 3.5 Sonnet for data science code generation"
)