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
from dotenv import load_dotenv
from io import StringIO, BytesIO
import seaborn as sns
from llm_sandbox import SandboxSession

from code_synth_agent import synthesize

import json
from datetime import datetime

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
# if 'conversation_filename' not in st.session_state: st.session_state.conversation_filename = None
if 'model_choice' not in st.session_state: st.session_state.model_choice = "groq-gemma"

# Title & description
st.title("Data Analysis LLM Agent")
st.markdown("""
This application uses a LangGraph‑powered agent to generate Python code
for data analysis. Code is executed inside an isolated sandbox to
capture outputs and figures securely.
""")

# Sidebar
with st.sidebar:
    st.header("Configuration")
    
    # API Key Status
    st.subheader("API Keys")
    groq_api_key = os.getenv("GROQ_API_KEY")
    anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
    openai_api_key = os.getenv("OPENAI_API_KEY")
    gemini_api_key = os.getenv("GOOGLE_API_KEY")
    
    if st.session_state.model_choice in ["groq-gemma", "groq-llama"]:
        if groq_api_key:
            st.success("Groq API Key: ✓ Connected")
        else:
            st.error("Groq API Key: ✗ Missing")
            st.info("Add GROQ_API_KEY to your .env file")
    
    if st.session_state.model_choice == "claude":
        if anthropic_api_key:
            st.success("Claude API Key: ✓ Connected")
        else:
            st.error("Claude API Key: ✗ Missing")
            st.info("Add ANTHROPIC_API_KEY to your .env file")
    
    if st.session_state.model_choice in ["openai-gpt4.1", "openai-o4mini"]:
        if openai_api_key:
            st.success("OpenAI API Key: ✓ Connected")
        else:
            st.error("OpenAI API Key: ✗ Missing")
            st.info("Add OPENAI_API_KEY to your .env file")
    
    if st.session_state.model_choice == "gemini-2.5":
        if gemini_api_key:
            st.success("Gemini API Key: ✓ Connected")
        else:
            st.error("Gemini API Key: ✗ Missing")
            st.info("Add GEMINI_API_KEY to your .env file")
        
    # Model selection
    st.subheader("Select Model")
    model_choice = st.selectbox(
        "Choose an LLM model:",
        [
            "groq-gemma", 
            "groq-llama", 
            "claude", 
            "openai-gpt4.1", 
            "openai-o4mini",
            "gemini-2.5"
        ],
        index=["groq-gemma", "groq-llama", "claude", "openai-gpt4.1", "openai-o4mini", "gemini-2.5"].index(st.session_state.model_choice),
        key="model_selector"
    )
    
    # Update session state when selection changes
    st.session_state.model_choice = model_choice
    
    # Data upload
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
    
    
    # st.subheader("Load Previous Conversation")

    # # Get list of saved sessions
    # session_files = sorted(
    #     [f for f in os.listdir(CONVO_DIR) if f.endswith('.json')],
    #     reverse=True
    # )

    # if session_files:
    #     selected_file = st.selectbox("Choose session to load", session_files)

    #     if st.button("Load Conversation"):
    #         print(f"Loading conversation from {selected_file}")
    #         try:
    #             with open(os.path.join(CONVO_DIR, selected_file), 'r') as f:
    #                 st.session_state.conversation = json.load(f)
    #                 st.session_state.conversation_filename = selected_file
    #                 st.success(f"Loaded: {selected_file}")
    #                 st.rerun()
    #                 print(f"loaded conversation - {st.session_state.conversation}")
    #         except Exception as e:
    #             st.error(f"Failed to load conversation: {e}")
    # else:
    #     st.info("No previous sessions found.")

    
    
    
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
import shutil
# Remove all files in /sandbox/figs if it exists
if os.path.exists('/sandbox/figs'):
    shutil.rmtree('/sandbox/figs')
os.makedirs('/sandbox/figs', exist_ok=True)
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
        
        st.subheader('Conversation')
    
        # Display chat messages from history on app rerun
        for i, message in enumerate(st.session_state.conversation):
            with st.chat_message(message["role"]):
                if message['role'] == 'user':
                    st.markdown(message["content"])
                else:
                    st.markdown(message["content"]['prefix'])
                    with st.expander('Show code'):
                        st.code(message['content']['code'], language='python')
                        b1, _, b2 = st.columns([2, 6, 2])
                        # Add download button for the code
                        with b1:
                            st.download_button(
                                label="💾 Save code",
                                data=message['content']['code'],
                                file_name=f"code_snippet_{i}.py",
                                mime="text/plain",
                                key=f"download_code_{i}"
                            )
                        with b2:
                            # Add run button for the code
                            if st.button("▶️ Run Code", key=f"run_code_{i}"):
                                st.session_state.generated_code = message['content']['code']
                                execute_code()
                        
        
        ui = st.chat_input("Ask about your data")
        if ui:
            with st.spinner(f'Generating code via {st.session_state.model_choice}…'):
                try:
                    out = synthesize(ui, st.session_state.dataset_info, st.session_state.dataset_path, st.session_state.model_choice)
                    st.session_state.generated_code = out['code'].strip()
                    st.session_state.conversation.append({'role':'user', 'content':ui, 'type':'text'})
                    st.session_state.conversation.append({'role':'assistant', 'content':{'prefix': out['prefix'], 
                                                                                         'code':st.session_state.generated_code
                                                                                         }, 'type':'text'})
                    # save_conversation_to_file()
                    if st.session_state.code_execution_results is not None:
                        st.session_state.code_execution_results['figures'] = []
                    execute_code()
                    st.rerun()
                except Exception as e:
                    st.error(f'Error generating code: {e}')
                    logger.error('Synthesis failed', exc_info=True)
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
            st.error('Execution failed'); st.error(r['error'])
        if r['output']: st.subheader('Output'); st.text(r['output'])
        if r['figures']:
            st.subheader('Visualizations')
            for i,b64 in enumerate(r['figures']): 
                st.image(BytesIO(base64.b64decode(b64)),caption=f'Figure {i+1}')
