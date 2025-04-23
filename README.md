
# CodeSynth

## Overview
CodeSynth is an AI-powered data analysis assistant that generates Python code for data analysis tasks using Large Language Models. Built on LangGraph and Streamlit, it provides an intuitive interface for users to interact with various LLMs to solve data analysis problems without writing code themselves.

## System Architecture

CodeSynth is built with a modular architecture that separates the user interface, code generation logic, and execution environment:

### Component Overview
- **Streamlit Web App** (`app.py`): The user interface for dataset selection and query input
- **Code Synthesis Agent** (`code_synth_agent.py`): The LangGraph-powered workflow for code generation
- **Execution Sandbox** (`llm-sandbox`): Secure environment for executing generated code
- **Agent Implementations** (`agents/`): Model-specific implementations for different LLM providers
- **Execution Engine** (`components/execution_engine.py`): Code execution with safety controls
- **Evaluation Framework** (`run_evaluation.py`): Benchmarking system for agent performance

### User Interface (app.py)
- Split-panel layout with dataset preview, conversation history, and code/output display
- Model selection for multiple LLM providers
- Dataset management (upload, sample selection)
- Conversation history with save/load functionality
- Visualization rendering and output display

### Code Generation (code_synth_agent.py)
- LangGraph workflow with generate and validate steps
- Structured output with Pydantic models
- Multi-model support with dynamic selection
- Automatic error detection and code correction
- Dataset context injection for relevant results

### Execution Environment
- Isolated sandbox with resource limits
- Pre-loaded data science libraries
- Figure capture and rendering
- Error handling and reporting

### How It Works

1. User enters a natural language query about their dataset
2. The query and dataset information are passed to the synthesis agent
3. The agent generates Python code using the selected LLM model
4. Generated code is validated by executing it in a sandbox environment
5. If errors occur, the agent attempts to fix the code automatically
6. The successful code and its output are displayed to the user
7. Generated visualizations are captured and rendered in the interface
8. The conversation history is updated with the user query and generated code

## Key Features
- Interactive Streamlit web interface for data analysis
- Support for multiple LLM providers (OpenAI, Anthropic Claude, Groq)
- Code generation for common data analysis tasks
- Secure code execution in an isolated sandbox environment
- Built-in sample datasets (Iris, Diamonds, Tips, etc.)
- Custom dataset upload (CSV, Excel, JSON)
- Conversation history saving and loading
- Visualization capabilities using matplotlib and seaborn
- Comprehensive evaluation framework for agent performance

## Installation

### Prerequisites
- Python 3.10+
- Docker (for sandbox execution environment)
- API keys for at least one LLM provider

### Setup
1. Clone the repository
```bash
git clone https://github.com/yourusername/codesynth.git
cd codesynth
```

2. Create and activate a virtual environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies
```bash
pip install -r requirements.txt
```

4. Create a `.env` file with your API keys
```
OPENAI_API_KEY=your_openai_key
ANTHROPIC_API_KEY=your_anthropic_key
GROQ_API_KEY=your_groq_key
```

5. Run the setup script to configure the sandbox environment
```bash
bash setup.sh
```

## Getting Started

Follow these steps to quickly begin analyzing data with CodeSynth:

### Quick Start

1. **Launch the application**
   ```bash
   streamlit run app.py
   ```

2. **Select a model**
   - Choose "claude" for high-quality analysis
   - Choose "groq-gemma" for faster response times
   - Choose "openai-gpt4.1" for complex analyses

3. **Load sample data**
   - Select "Iris" from the sample datasets dropdown
   - The dataset preview will appear in the main panel

4. **Enter your first query**
   - Try: "Create a scatter plot matrix of the iris dataset with colors by species"
   - The agent will generate and execute the code automatically
   - View the results in the right panel

5. **Iterate on your analysis**
   - Ask follow-up questions about the data
   - Request modifications to the visualization
   - Download generated code for further customization


## Troubleshooting

### API Keys Issues
- **Missing API Keys**: Ensure your `.env` file contains the necessary API keys
- **Authentication Errors**: Verify API key validity and check for whitespace or formatting issues
- **Rate Limiting**: If you encounter "Rate limit exceeded" errors, wait a few minutes or switch to a different model provider

### Sandbox Execution Problems
- **Missing Dependencies**: If code fails with `ImportError`, check the allowed libraries list in `config.yaml`
- **Execution Timeouts**: For complex analyses, increase the `timeout` value in `config.yaml`
- **Memory Errors**: Increase `max_memory_mb` in `config.yaml` for larger datasets

### Dataset Handling
- **Encoding Issues**: If text appears corrupted, try manually specifying the encoding when uploading
- **Large Datasets**: For datasets over 100MB, increase `max_upload_size_mb` in configuration
- **Unsupported Formats**: Convert non-standard formats to CSV before uploading

### Common Error Messages
- **"Error loading dataset"**: Check if the file is corrupted or in an unsupported format
- **"Execution failed"**: Review the error message for syntax errors or missing libraries
- **"No visualizations generated"**: Ensure your query explicitly requests a visualization

## Usage
1. Start the Streamlit application
```bash
streamlit run app.py
```

2. In the web interface:
   - Select an LLM model from the sidebar
   - Upload your dataset or choose a sample dataset
   - Enter your data analysis question in natural language
   - Review the generated code and results

### Example Queries
- "Create a scatter plot showing the relationship between sepal length and width"
- "Perform a cluster analysis on the iris dataset and visualize the results"
- "Calculate summary statistics for each species in the dataset"
- "Build a regression model to predict diamond prices based on their features"

## Evaluation Framework

The system includes a comprehensive evaluation framework for benchmarking agent performance across multiple dimensions:

### Custom Evaluation Metrics

The evaluation system uses a weighted scoring approach with five key metrics:

- **Functional Correctness (35%)**: Measures if code executes without errors and produces expected outputs
- **Code Quality (20%)**: Assesses structural elements including:
  - Comment ratio
  - Proper imports
  - Error handling
  - Function definitions
- **Query Relevance (25%)**: Evaluates how well the generated code addresses the user's prompt
- **Execution Metrics (10%)**: Measures runtime performance, memory usage, and efficiency
- **Visualization Quality (10%)**: Evaluates the quality and appropriateness of data visualizations

Each metric is scored from 0.0 to 1.0, with the weighted combination producing an overall score for each test case.

### Test Suite

The evaluation framework includes a structured test suite with:

- 7 test categories (basic_statistics, data_cleaning, correlation_analysis, etc.)
- 4 test queries per category with varying complexity
- Pre-configured dataset mappings for each category
- Standardized test case generation

### Running Evaluations

```bash
python run_evaluation.py --agent <agent_id> --categories "basic_statistics,visualization" --limit 10
```

### Results Analysis

The framework includes:
- JSON result storage in `evaluation/results/`
- Comparative performance visualization across agents
- Detailed per-test-case metrics and code analysis
- Token efficiency metrics for cost analysis

To generate comparison visualizations across multiple agents:

```bash
python run_evaluation.py --compare --agents "openai-gpt4.1,claude-sonnet" --categories "visualization,advanced_analysis"
```


### DS-1000 Benchmark Adaptation

CodeSynth incorporates a custom adaptation of the [DS-1000 benchmark](https://ds1000-code-gen.github.io/), a prominent academic benchmark for data science code generation:

- **Curated Selection**: We've carefully selected 60 representative problems from the original DS-1000 benchmark
- **Custom Integration**: Modified the evaluation framework to work seamlessly with our LLM agents
- **Direct Comparison**: Enables comparison with published academic results while being computationally manageable

#### About DS-1000

DS-1000 is a comprehensive benchmark created by Lai et al. (2022) featuring 1,000 data science problems that:
- Reflect realistic use cases collected from StackOverflow
- Include reliable metrics for functional correctness
- Guard against model memorization through problem perturbations
- Span seven popular Python libraries (NumPy, Pandas, Matplotlib, etc.)

For more information, including the original paper, code, and dataset, visit the [official DS-1000 website](https://ds1000-code-gen.github.io/).

#### Implementation Details

- **Problem Selection**: 60 problems distributed evenly across core data science libraries:
  - **Pandas**: 15 problems testing data manipulation and transformation
  - **NumPy**: 15 problems focusing on array operations
  - **Matplotlib**: 15 problems evaluating visualization capabilities
  - **SciPy**: 15 problems covering scientific computing functions

- **Evaluation Process**:
  - Problems are presented to models with minimal context
  - Solutions are evaluated using the original DS-1000 test harness
  - Isolated execution ensures accurate performance measurement
  - Results are categorized by library and problem type

#### LLM-Specific Adaptations

The benchmark runner is customized for each LLM with optimized prompting:
```python
# Example for Claude 3.5 Sonnet
response = client.messages.create(
    model="claude-3-5-sonnet-20240620",
    system="Write a short code following the given format and indentation. Only provide the exact code completion needed.",
    max_tokens=1024,
    temperature=0,
    messages=[{"role": "user", "content": prompt}]
)
```

#### Running the DS-1000 Evaluation

```bash
python run_evaluation.py --benchmark ds1000 --agent <agent_id>
# Or for the direct runner:
python DS-1000-main/selected_problems_60/run_claude_selected.py  # For Claude 3.5
```

Results are analyzed by library category and aggregated for comprehensive performance assessment, allowing us to identify strengths and weaknesses across different data science domains.


## Project Structure
```
CodeSynth/
├── agents/               # Agent implementations
│   ├── base_agent.py     # Abstract base class for all agents
│   ├── openai_agent.py   # OpenAI-specific implementation
│   ├── claude_agent.py   # Claude-specific implementation
│   ├── groq_agent.py     # Groq-specific implementation
│   └── opensource_agent.py # Open source model implementation
├── app.py                # Main Streamlit application
├── code_synth_agent.py   # LangGraph agent implementation
├── components/           # UI components
│   └── execution_engine.py # Secure code execution environment
├── config/               # Configuration files
├── data/                 # Data directory
│   ├── conversations/    # Saved conversation history
│   ├── sample_datasets/  # Built-in datasets
│   └── user_datasets/    # User-uploaded datasets
├── database/             # Database utilities
├── debug_app.py          # Debugging version of app
├── DS-1000-main/         # DS-1000 benchmark adaptation
│   ├── selected_problems_60/ # Curated subset of DS-1000 problems + implementation to run on selected subset
├── evaluation/           # Evaluation framework
├── multi_agent_synth.py  # Multi-agent system
├── requirements.txt      # Python dependencies
├── run_evaluation.py     # Evaluation script
├── run.sh                # Launch script
├── setup.sh              # Setup script
└── utils/                # Utility function
```


## Agent Framework

The `agents/` directory contains model-specific implementations:

### Base Agent
The `base_agent.py` defines the abstract interface all agents implement:
- Configuration and context management
- Methods for code generation, Q&A, and code improvement
- Dataset formatting utilities

### Model-Specific Agents
- **OpenAI Agent** (`openai_agent.py`): Implementation for GPT-4.1 and o4-mini models
- **Claude Agent** (`claude_agent.py`): Implementation for Anthropic's Claude-3-5-sonnet
- **Groq Agent** (`groq_agent.py`): Support for gemma2-9b-it and llama-3.3-70b-versatile
- **Open Source Agent** (`opensource_agent.py`): Implementation for open source models

## Execution Engine

The `components/execution_engine.py` provides secure code execution:

### Features
- Secure sandbox environment
- Resource limits (time, memory)
- Pre-loaded data science libraries
- Output and figure capture
- Dataset loading with encoding detection

### MLSandbox
A specialized class for machine learning operations with additional safety checks.



## Configuration

The `config/config.yaml` file contains the main configuration settings for CodeSynth:

### Application Settings
- **Name**: Data Analysis LLM Agent
- **Debug Mode**: Configurable via DEBUG environment variable
- **Log Level**: Customizable through LOG_LEVEL environment variable

### Model Configuration
- **OpenAI**: 
  - Default model: gpt-4.1
  - Max tokens: 1024
  - Temperature: 0.2
  - Timeout: 60 seconds

- **Claude**:
  - Default model: claude-3-5-sonnet-20240620
  - Max tokens: 4096
  - Temperature: 0.2
  - Timeout: 60 seconds

- **Open Source Models**:
  - Supported models: phi3-mini-4k, phi3-mini-128k
  - Configurable via environment variables
  - Higher timeout (120 seconds) for local inference

### Execution Engine
- **Code Execution Timeout**: 30 seconds
- **Memory Limit**: 1024 MB
- **Sandbox Type**: llm_sandbox (alternative: restricted_python)
- **Allowed Libraries**: pandas, numpy, matplotlib, seaborn, plotly, scikit-learn, scipy, statsmodels

### Dataset Management
- **Sample Datasets Path**: data/sample_datasets
- **User Datasets Path**: data/user_datasets
- **Maximum Upload Size**: 100 MB

## Utilities

The `utils/` directory contains utility functions that support the core functionality:

### Prompt Templates

The `prompt_templates.py` module provides structured prompts for different LLM providers:

- **LLM-Specific Templates**: Optimized prompts for each supported model:
  - OpenAI templates for GPT-4.1
  - Claude templates for Claude-3.5-sonnet
  - Templates for other supported models

- **Template Categories**:
  - `code_generation`: Templates for generating Python data analysis code
  - `question_answering`: Templates for answering data science questions
  - `code_improvement`: Templates for improving and fixing generated code

- **Template Features**:
  - Dataset information formatting
  - Error handling instructions
  - Encoding detection for data loading
  - Standardized formatting for each model's preferred input structure
  - Library usage constraints and best practices

The template system ensures consistent, high-quality outputs across different LLM providers while leveraging each model's strengths.



## Expanded details for Core Implementation

### Streamlit Application (app.py)

The main interface of CodeSynth is built with Streamlit and provides:

- **Interactive UI**: Split-panel layout with dataset preview, conversation history, and code/output display
- **Model Selection**: Support for multiple LLMs including OpenAI, Claude, and Groq models
- **Dataset Management**:
  - Upload custom datasets (CSV, Excel, JSON)
  - Choose from built-in sample datasets (Iris, Diamonds, Tips, Planets)
- **Conversation Management**:
  - Persistent chat interface with history
  - Save/load conversation functionality
  - Individual code execution from conversation history
- **Secure Execution**:
  - Isolated sandbox environment with llm-sandbox
  - Figure capture and rendering
  - Execution metrics and error reporting

### Code Synthesis Agent (code_synth_agent.py)

The LangGraph-powered agent orchestrates the code generation process:

- **LangGraph Workflow**:
  - `generate`: Creates Python code based on user query and dataset
  - `code_check`: Validates generated code by execution in sandbox
  - Automatic retry mechanism for failed generations (up to 3 attempts)
- **Structured Output**: Uses Pydantic models to ensure consistent code format
- **Multi-Model Support**: Dynamically selects the appropriate LLM based on user choice
- **Enhanced Prompting**: Specialized prompts with dataset context and execution environment details
- **Error Handling**: Detects execution failures and tries to correct code issues
- **Sandbox Integration**: Secure execution in isolated environment with predefined libraries
