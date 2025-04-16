"""
Simplified Streamlit app for debugging Claude 3.5 integration
"""

import os
import logging
import time
import yaml
import pandas as pd
import streamlit as st
import anthropic
from dotenv import load_dotenv

# Configure logging - more verbose to catch issues
logging.basicConfig(
    level=logging.DEBUG,
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
    page_title="Data Analysis LLM Agent - Debug Version",
    page_icon="🔍",
    layout="wide"
)

# Initialize session state
if 'dataset' not in st.session_state:
    st.session_state.dataset = None
if 'dataset_info' not in st.session_state:
    st.session_state.dataset_info = {}
if 'generated_code' not in st.session_state:
    st.session_state.generated_code = ""

# App title and description
st.title("Data Analysis LLM Agent - Debug Version")
st.markdown("Simplified version to debug Claude 3.5 integration")

# Sidebar
with st.sidebar:
    st.header("Dataset")
    
    # Sample dataset selection
    sample_dataset = st.selectbox(
        "Select a sample dataset:",
        ["Iris"]
    )
    
    if st.button("Load Dataset"):
        try:
            # Use sklearn's built-in iris dataset
            from sklearn.datasets import load_iris
            iris = load_iris()
            iris_df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
            iris_df['species'] = pd.Categorical.from_codes(iris.target, iris.target_names)
            
            st.session_state.dataset = iris_df
            
            # Extract dataset information
            st.session_state.dataset_info = {
                "name": "Iris Dataset",
                "shape": iris_df.shape,
                "columns": [{"name": col, "type": str(iris_df[col].dtype)} for col in iris_df.columns]
            }
            
            st.success("Iris dataset loaded successfully!")
        except Exception as e:
            st.error(f"Error loading dataset: {e}")
            logger.error(f"Error loading dataset: {e}", exc_info=True)

# Main area
if st.session_state.dataset is not None:
    st.subheader("Dataset Preview")
    st.dataframe(st.session_state.dataset.head())
    
    st.subheader("Generate Code")
    user_query = st.text_area(
        "What would you like to analyze?",
        "Create a scatter plot of sepal length vs. sepal width colored by species."
    )
    
    if st.button("Generate Code", type="primary"):
        try:
            logger.debug("Starting code generation...")
            
            # Initialize Anthropic client
            api_key = os.getenv("ANTHROPIC_API_KEY")
            if not api_key:
                st.error("ANTHROPIC_API_KEY not found in environment variables.")
                logger.error("ANTHROPIC_API_KEY not found in environment variables.")
            else:
                logger.debug("API key found, initializing Anthropic client...")
                client = anthropic.Anthropic(api_key=api_key)
                
                # Format the prompt
                dataset_info_text = f"""
                Dataset name: {st.session_state.dataset_info['name']}
                Shape: {st.session_state.dataset_info['shape'][0]} rows, {st.session_state.dataset_info['shape'][1]} columns
                Columns: {', '.join([col['name'] for col in st.session_state.dataset_info['columns']])}
                """
                
                prompt = f"""
                Generate Python code for the following data analysis task:
                
                User query: {user_query}
                
                Dataset information:
                {dataset_info_text}
                
                The dataset is already loaded into a pandas DataFrame called 'df'.
                Generate only Python code, no explanations needed.
                """
                
                logger.debug("Sending request to Anthropic API...")
                start_time = time.time()
                
                # Use a simple timeout mechanism
                try:
                    response = client.messages.create(
                        model="claude-3-5-sonnet-20240620",
                        max_tokens=1000,
                        temperature=0.2,
                        messages=[{"role": "user", "content": prompt}]
                    )
                    logger.debug(f"Response received in {time.time() - start_time:.2f} seconds")
                    
                    # Process the response
                    code_text = response.content[0].text
                    
                    # Extract code if it's wrapped in markdown
                    if "```python" in code_text:
                        code_parts = code_text.split("```python")
                        if len(code_parts) > 1:
                            code_text = code_parts[1].split("```")[0]
                    
                    st.session_state.generated_code = code_text.strip()
                    logger.debug("Code generated successfully!")
                    
                except Exception as e:
                    st.error(f"Error during API request: {e}")
                    logger.error(f"Error during API request: {e}", exc_info=True)
            
        except Exception as e:
            st.error(f"Error generating code: {e}")
            logger.error(f"Error generating code: {e}", exc_info=True)
    
    # Display generated code
    if st.session_state.generated_code:
        st.subheader("Generated Code")
        st.code(st.session_state.generated_code, language="python")
        
        # Simple execution capability
        if st.button("Run Code"):
            try:
                # Prepare environment for execution
                locals_dict = {"df": st.session_state.dataset}
                
                # For displaying plots in Streamlit
                exec("import matplotlib.pyplot as plt", locals_dict)
                exec("import seaborn as sns", locals_dict)
                
                # Execute the code
                exec(st.session_state.generated_code, locals_dict)
                
                # If there's a plot, display it
                if 'plt' in locals_dict:
                    st.pyplot(locals_dict.get('plt').gcf())
                    plt = locals_dict.get('plt')
                    plt.close()
            except Exception as e:
                st.error(f"Error executing code: {e}")
                logger.error(f"Error executing code: {e}", exc_info=True)
else:
    st.info("Please load a dataset from the sidebar first.")