"""
Enhanced Streamlit app for data analysis with LangGraph synthesis
(using llm-sandbox for code execution)
"""

import os
import shutil
import time
import base64
import pandas as pd
import numpy as np
import streamlit as st
from dotenv import load_dotenv
from io import StringIO, BytesIO
import matplotlib.pyplot as plt
import seaborn as sns
from llm_sandbox import SandboxSession

from code_synth_agent import synthesize

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
This application uses a LangGraph‑powered agent to generate Python code
for data analysis. Code is executed inside an isolated sandbox to
capture outputs and figures securely.
""")

# Sidebar
with st.sidebar:
    st.header("Configuration")
    api_key = os.getenv("GROQ_API_KEY")
    if api_key:
        st.success("LangGraph API Key: ✓ Connected")
    else:
        st.error("LangGraph API Key: ✗ Missing")
        st.info("Add GROQ_API_KEY to your .env file")
    # Data upload
    st.subheader("Upload Dataset")
    uploaded = st.file_uploader("Choose a CSV/Excel/JSON file", type=["csv","xlsx","xls","json"])
    if uploaded:
        try:
            save_dir = os.path.join("data","user_datasets"); os.makedirs(save_dir,exist_ok=True)
            path = os.path.join(save_dir, "data.csv")
            with open(path,'wb') as f: f.write(uploaded.getbuffer())
            if uploaded.name.endswith('csv'):
                df = pd.read_csv(path)
            elif uploaded.name.endswith(('xls','xlsx')):
                df = pd.read_excel(path)
            elif uploaded.name.endswith('json'):
                df = pd.read_json(path)
            else:
                df=None; st.error("Unsupported format")
            if df is not None:
                st.session_state.dataset = df
                st.session_state.dataset_path = path
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
    if sel!="None":
        try:
            if sel=="Iris":
                from sklearn.datasets import load_iris
                ir=load_iris(); df=pd.DataFrame(ir.data,columns=ir.feature_names)
                df['species']=pd.Categorical.from_codes(ir.target,ir.target_names)
                fname="iris.csv"
            else:
                df=sns.load_dataset(sel.lower()); fname=f"{sel.lower()}.csv"
            sd=os.path.join("data","sample_datasets"); os.makedirs(sd,exist_ok=True)
            fp=os.path.join(sd,fname); df.to_csv(fp,index=False)
            st.session_state.dataset, st.session_state.dataset_path = df, fp
            st.session_state.dataset_info={"name":sel,"shape":df.shape,
                "columns":[{"name":c,"type":str(df[c].dtype),"description":"","sample":str(df[c].iloc[0]) if not df.empty else ""} for c in df.columns],
                "sample":df.head(5).to_string()
            }
            st.success(f"Loaded sample: {sel}"); st.write(f"Shape: {df.shape[0]}×{df.shape[1]}")
        except Exception as e:
            st.error(f"Error loading sample: {e}")
    if st.button("Clear Conversation"):
        st.session_state.conversation=[]
        st.session_state.generated_code=""
        st.session_state.code_execution_results=None
        st.success("Cleared!")

# Execute code in sandbox

def execute_code():
    if not st.session_state.generated_code:
        return
    res={"success":False,"output":"","error":"","figures":[],"execution_time":0}
    buf=StringIO()
    start_time=time.time()
    try:
        # prepare temp dirs and script
        tmp_script=os.path.join("data","tmp_script.py")
        tmp_fig_dir=os.path.join("data","tmp_figs")
        if os.path.exists(tmp_fig_dir): shutil.rmtree(tmp_fig_dir)
        os.makedirs(tmp_fig_dir, exist_ok=True)
        # build script: monkey-patch, user code, save figs
        prelude="""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import base64
from io import BytesIO, StringIO
from sklearn import datasets, metrics, model_selection, preprocessing, decomposition, cluster
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
#         save_block="""
# # save figures to disk
# import os
# os.makedirs('/sandbox/figs',exist_ok=True)
# for i,fig in enumerate(_figs): fig.savefig(f'/sandbox/figs/fig{i}.png',dpi=100)
# """
        script_content = prelude + "\n" + st.session_state.generated_code + "\n"
        # script_content = prelude + "\n" + st.session_state.generated_code + "\n" + save_block
        with open(tmp_script,'w') as f: f.write(script_content)
        # run in sandbox
        with SandboxSession(lang='python', keep_template=True) as sess:
            # copy dataset and script
            sess.copy_to_runtime(st.session_state.dataset_path, '/sandbox/data.csv')
            sess.copy_to_runtime(tmp_script, '/sandbox/tmp_script.py')
            # execute
            result = sess.execute_command('python /sandbox/tmp_script.py')
            out = getattr(result,'text','')
            # collect figures
            # ls = sess.execute_command('ls -al /sandbox')
            # print(ls, ls.text)
            # ls = sess.execute_command('ls /sandbox/figs')
            # print(ls, ls.text)
            # names = ls.text.strip().split() if ls.text else []
            figs=[]
            # for fn in names:
            #     remote=f'/sandbox/figs/{fn}'
            #     local=os.path.join(tmp_fig_dir,fn)
            #     sess.copy_from_runtime(remote, local)
            #     with open(local,'rb') as im: figs.append(base64.b64encode(im.read()).decode('utf-8'))
        # populate results
        res['success']=True
        res['output']=out
        res['figures']=figs
    except Exception as e:
        res['error']=f"{type(e).__name__}: {e}"
    finally:
        res['execution_time']=time.time()-start_time
        st.session_state.code_execution_results=res

# Main layout
col1,col2=st.columns([3,2])
with col1:
    if st.session_state.dataset is not None:
        st.subheader('Dataset Preview')
        st.dataframe(st.session_state.dataset.head(),use_container_width=True)
        st.subheader('Ask About Your Data')
        ui=st.text_area('Enter your query:',height=100)
        b1,b2,b3,b4=st.columns(4)
        with b1: gen=st.button('Generate Code')
        with b2: run=st.button('Run Code',disabled=not st.session_state.generated_code)
        with b3: save=st.button('Save Code',disabled=not st.session_state.generated_code)
        with b4: new=st.button('New Query')
        if gen and ui:
            with st.spinner('Generating code via LangGraph…'):
                try:
                    out=synthesize(ui, st.session_state.dataset_info, st.session_state.dataset_path)
                    st.session_state.generated_code=out['code'].strip()
                    st.session_state.conversation.append({'role':'assistant','content':out['prefix'],'type':'text'})
                    st.session_state.conversation.append({'role':'assistant','content':st.session_state.generated_code,'type':'code'})
                    execute_code()
                except Exception as e:
                    st.error(f'Error generating code: {e}')
                    logger.error('Synthesis failed',exc_info=True)
        if run: execute_code()
        if save:
            dld=os.path.join('data','downloads'); os.makedirs(dld,exist_ok=True)
            fn=f"analysis_{time.strftime('%Y%m%d-%H%M%S')}.py"; path=os.path.join(dld,fn)
            with open(path,'w') as f: f.write(st.session_state.generated_code)
            with open(path) as f: ct=f.read()
            st.download_button('Download Python File',data=ct,file_name=fn)
            st.success(f'Saved as {fn}')
        if new: st.session_state.generated_code=st.session_state.code_execution_results=None
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

st.subheader('Conversation History')
for m in st.session_state.conversation:
    if m['role']=='user': st.markdown(f"**You:** {m['content']}")
    else:
        if m.get('type')=='code':
            st.markdown('**Assistant:** Generated code:')
            with st.expander('Show code'): st.code(m['content'],language='python')
        else: st.markdown(f"**Assistant:** {m['content']}")