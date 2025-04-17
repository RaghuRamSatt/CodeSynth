"""
Enhanced Streamlit app for data analysis with LangGraph synthesis
(using llm-sandbox for code execution)
"""

import os
import shutil
import time
import base64
import pandas as pd
import streamlit as st
import anthropic
import matplotlib.pyplot as plt
import seaborn as sns
from dotenv import load_dotenv
from io import StringIO, BytesIO
from groq import Groq
from openai import OpenAI
from utils.get_api_response import fetch_api_response

# Configure logging
import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment
load_dotenv()
st.set_page_config(page_title="Data Analysis LLM Agent", page_icon="📊", layout="wide")

# Session state init
if 'conversation' not in st.session_state: st.session_state.conversation = []
if 'dataset' not in st.session_state: st.session_state.dataset = None
if 'dataset_info' not in st.session_state: st.session_state.dataset_info = {}
if 'dataset_path' not in st.session_state: st.session_state.dataset_path = None
if 'generated_code' not in st.session_state: st.session_state.generated_code = ""
if 'code_execution_results' not in st.session_state: st.session_state.code_execution_results = None

# Title & description
st.title("Data Analysis LLM Agent")
st.markdown("""
This application uses Generative AI models to generate Python code for data analysis based on your natural language queries. 
Simply load a dataset, ask questions about your data, and get Python code and visualizations.
""")

# Sidebar
with st.sidebar:
    st.header("Configuration")
    
    # API key status
    ai_options = ["Anthropic", "Open AI", "Llama", "Other"]
    selected_ai_option = st.selectbox("Choose an AI Agent: ", ai_options)
    ai_options_dict = {"Anthropic": "ANTHROPIC_API_KEY", "Open AI": "OPENAI_API_KEY", "Llama":"GROQ_API_KEY", "Other":"OTHER_API_KEY"}
    api_key = os.getenv(ai_options_dict[selected_ai_option])
    if api_key:
        st.success(f"{selected_ai_option} API Key: ✓ Connected")
    else:
        st.error(f"{selected_ai_option} API Key: ✗ Missing")
        st.info(f"Add {ai_options_dict[selected_ai_option]} to your .env file")
    
    # Dataset upload
    st.subheader("Upload Dataset")
    uploaded = st.file_uploader("Choose a CSV/Excel/JSON file", type=["csv","xlsx","xls","json"])
    if uploaded:
        try:
            save_dir = os.path.join("data", "user_datasets")
            os.makedirs(save_dir, exist_ok=True)
            path = os.path.join(save_dir, uploaded.name)
            with open(path,'wb') as f: f.write(uploaded.getbuffer())
            if uploaded.name.endswith('csv'):
                df = pd.read_csv(path)
            elif uploaded.name.endswith(('xls','xlsx')):
                df = pd.read_excel(path)
            elif uploaded.name.endswith('json'):
                df = pd.read_json(path)
            else:
                df = None; st.error("Unsupported format")
            df.to_csv(os.path.join(save_dir, 'data.csv'), index=False)
            if df is not None:
                st.session_state.dataset = df
                st.session_state.dataset_path = os.path.join(save_dir, 'data.csv')
                st.session_state.dataset_info = {
                    "name": uploaded.name,
                    "shape": df.shape,
                    "columns": [
                        {"name":c,"type":str(df[c].dtype),"description":"","sample":str(df[c].iloc[0]) if not df.empty else ""}
                        for c in df.columns
                    ],
                    "sample": df.head(5).to_string()
                }
                st.success(f"Loaded: {uploaded.name}")
                st.write(f"Shape: {df.shape[0]} rows × {df.shape[1]} cols")
        except Exception as e:
            st.error(f"Error loading dataset: {e}")
    # Sample data
    st.subheader("Or Use Sample Dataset")
    choices=["None","Iris","Diamonds","Tips","Planets"]
    sel=st.selectbox("Select sample:",choices)
    if sel != "None":
        try:
            fname = "data.csv"
            if sel == "Iris":
                from sklearn.datasets import load_iris
                ir = load_iris()
                df = pd.DataFrame(ir.data, columns=ir.feature_names)
                df['species'] = pd.Categorical.from_codes(ir.target, ir.target_names)
            else:
                df = sns.load_dataset(sel.lower())
            sd = os.path.join("data", "sample_datasets")
            os.makedirs(sd, exist_ok=True)
            fp = os.path.join(sd, fname)
            df.to_csv(fp, index=False)
            st.session_state.dataset, st.session_state.dataset_path = df, fp
            st.session_state.dataset_info={"name":sel,"shape":df.shape,
                "columns":[{"name":c,"type":str(df[c].dtype),"description":"","sample":str(df[c].iloc[0]) if not df.empty else ""} for c in df.columns],
                "sample":df.head(5).to_string()
            }
            st.success(f"Loaded sample: {sel}"); st.write(f"Shape: {df.shape[0]}×{df.shape[1]}")
        except Exception as e:
            st.error(f"Error loading sample: {e}")
    if st.button("Clear Conversation"):
        st.session_state.conversation = []
        st.session_state.generated_code = ""
        st.session_state.code_execution_results = None
        st.success("Cleared!")

# Execute code in sandbox

def execute_code():
    if not st.session_state.generated_code:
        return
    res = {"success":False, "output":"", "error":"", "figures":[], "execution_time":0}
    start_time = time.time()
    try:
        # prepare temp dirs and script
        tmp_script = os.path.join("data", "tmp_script.py")
        tmp_fig_dir = os.path.join("data", "tmp_figs")
        if os.path.exists(tmp_fig_dir): shutil.rmtree(tmp_fig_dir)
        os.makedirs(tmp_fig_dir, exist_ok=True)
        # build script: monkey-patch, user code, save figs
        prelude="""
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
_original_figure=plt.figure
_figs=[]
def _cf(*a,**k):
    fig=_original_figure(*a,**k)
    _figs.append(fig)
    return fig
plt.figure=_cf
_orig_sub=plt.subplots
def _cs(*a,**k):
    fig,ax=_orig_sub(*a,**k)
    _figs.append(fig)
    return fig,ax
plt.subplots=_cs
"""
        save_block="""
# save figures to disk
import os
os.makedirs('/sandbox/figs',exist_ok=True)
for i,fig in enumerate(_figs): fig.savefig(f'/sandbox/figs/fig{i}.png',dpi=100)
"""
        script_content = prelude + "\n" + st.session_state.generated_code + "\n" + save_block
        with open(tmp_script,'w') as f: f.write(script_content)
        # run in sandbox
        with SandboxSession(lang='python', keep_template=True) as sess:
            # copy dataset and script
            sess.copy_to_runtime(st.session_state.dataset_path, '/sandbox/data.csv')
            sess.copy_to_runtime(tmp_script, '/sandbox/tmp_script.py')
            # execute
            result = sess.execute_command('python /sandbox/tmp_script.py')
            out = getattr(result, 'text', '')
            # collect figures
            ls = sess.execute_command('ls /sandbox/figs')
            names = ls.text.strip().split() if ls.text else []
            figs = []
            for fn in names:
                remote = f'/sandbox/figs/{fn}'
                local = os.path.join(tmp_fig_dir,fn)
                sess.copy_from_runtime(remote, local)
                with open(local,'rb') as im: figs.append(base64.b64encode(im.read()).decode('utf-8'))
        # populate results
        res['success'] = True
        res['output'] = out
        res['figures'] = figs
    except Exception as e:
        res['error'] = f"{type(e).__name__}: {e}"
    finally:
        res['execution_time'] = time.time()-start_time
        st.session_state.code_execution_results=res

# Main layout
col1, col2 = st.columns([3,2])
with col1:
    
    if st.session_state.dataset is not None:
    
        st.subheader('Dataset Preview')
        st.dataframe(st.session_state.dataset.head(),use_container_width=True)
        
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
                with st.spinner(f"Generating code with {selected_ai_option}..."):
                    # Check for API key
                    # api_key = os.getenv("ANTHROPIC_API_KEY")
                    api_key = os.getenv(ai_options_dict[selected_ai_option])
                    if not api_key:
                        st.error(f"{ai_options_dict[selected_ai_option]} not found in environment variables.")
                        logger.error(f"{ai_options_dict[selected_ai_option]} not found in environment variables.")
                    else:
                        # Initialize AI client
                        if selected_ai_option == "Anthropic":
                            client = anthropic.Anthropic(api_key=api_key)
                        elif selected_ai_option == "Llama":
                            client = Groq(api_key=api_key)
                        elif selected_ai_option == "Open AI":
                            client = OpenAI(api_key=api_key)
                        elif selected_ai_option == "Other":
                            client = Groq(api_key=api_key)
                        
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
                        
                        # Send request to the chosen API
                        start_time = time.time()
                        code_text = fetch_api_response(ai_model=selected_ai_option, client=client, prompt=prompt)
                        
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
        st.info('Please load a dataset first.')

    
    
with col2:
    st.subheader('Generated Code')
    if st.session_state.generated_code: st.code(st.session_state.generated_code,language='python')
    else: st.info('Code will appear here.')
    if st.session_state.code_execution_results:
        r=st.session_state.code_execution_results
        if r['success']:
            st.success('Code executed successfully')
            st.write(f"Execution time: {r['execution_time']:.2f}s")
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
    f"Data Analysis LLM Agent - Using {selected_ai_option} for data science code generation"
)