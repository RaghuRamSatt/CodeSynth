#!/bin/bash
# Setup script for Data Analysis LLM Agent
# This script sets up the project environment and dependencies

set -e  # Exit on error

# Color output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
NC='\033[0m' # No Color

# Print header
echo -e "${GREEN}=====================================${NC}"
echo -e "${GREEN}   Data Analysis LLM Agent Setup     ${NC}"
echo -e "${GREEN}=====================================${NC}"

# Check Python version
echo -e "\n${YELLOW}Checking Python version...${NC}"
python_version=$(python --version)
if [[ $python_version == *"Python 3"* ]]; then
    echo -e "${GREEN}Python is installed: $python_version${NC}"
else
    echo -e "${RED}Python 3 is required but not found.${NC}"
    echo -e "${YELLOW}Please install Python 3 and try again.${NC}"
    exit 1
fi

# Create project directory structure
echo -e "\n${YELLOW}Creating project directory structure...${NC}"
mkdir -p data-analysis-llm-agent
cd data-analysis-llm-agent
mkdir -p agents components database utils evaluation data/sample_datasets data/user_datasets tests config

# Create virtual environment
echo -e "\n${YELLOW}Creating virtual environment...${NC}"
python -m venv venv
source venv/bin/activate

# Install dependencies
echo -e "\n${YELLOW}Installing dependencies...${NC}"
pip install --upgrade pip
pip install streamlit langchain langgraph openai anthropic pandas numpy matplotlib seaborn plotly scikit-learn python-dotenv llm-sandbox mysql-connector-python sqlalchemy transformers torch requests huggingface_hub

# Download sample datasets
echo -e "\n${YELLOW}Downloading sample datasets...${NC}"
mkdir -p data/sample_datasets

# Superstore dataset
echo "Downloading Superstore dataset..."
wget -O data/sample_datasets/superstore.csv https://raw.githubusercontent.com/tableau/sample-data/master/SuperstoreSales.csv 2>/dev/null || {
    echo -e "${RED}Failed to download Superstore dataset${NC}"
    echo -e "${YELLOW}You may need to manually download it later${NC}"
}

# S&P 500 dataset
echo "Downloading S&P 500 dataset..."
wget -O data/sample_datasets/sp500.csv https://raw.githubusercontent.com/datasets/s-and-p-500-companies/master/data/constituents.csv 2>/dev/null || {
    echo -e "${RED}Failed to download S&P 500 dataset${NC}"
    echo -e "${YELLOW}You may need to manually download it later${NC}"
}

# Marketing Campaign dataset
echo "Downloading Marketing Campaign dataset..."
wget -O data/sample_datasets/marketing_campaign.csv https://raw.githubusercontent.com/rodsaldanha/marketing-campaign/master/marketing_campaign.csv 2>/dev/null || {
    echo -e "${RED}Failed to download Marketing Campaign dataset${NC}"
    echo -e "${YELLOW}You may need to manually download it later${NC}"
}

# Create .env file template
echo -e "\n${YELLOW}Creating .env file template...${NC}"
cat > .env << EOF
# API Keys
OPENAI_API_KEY=your_openai_key_here
ANTHROPIC_API_KEY=your_anthropic_key_here

# Database Configuration
MYSQL_HOST=localhost
MYSQL_USER=root
MYSQL_PASSWORD=your_password_here
MYSQL_DATABASE=data_analysis_agent

# Model Configuration
# For local model paths if using Phi-3 or Mistral locally
PHI3_MODEL_PATH=/path/to/phi3/model
MISTRAL_MODEL_PATH=/path/to/mistral/model

# Alternatively, if using hosted models
PHI3_ENDPOINT=your_endpoint_if_using_hosted_phi3
MISTRAL_ENDPOINT=your_endpoint_if_using_hosted_mistral

# App Configuration
DEBUG=False
LOG_LEVEL=INFO
MAX_TOKENS=8192
CONTEXT_WINDOW=8192
EOF

echo -e "${GREEN}Created .env template. Please edit with your API keys.${NC}"

# Create README file
echo -e "\n${YELLOW}Creating README file...${NC}"
cat > README.md << EOF
# Data Analysis LLM Agent

This project implements an AI-powered application that compares different LLM models for generating Python code for data analysis tasks.

## Features

- Multiple LLM model support (GPT-4, Claude 3.5, Phi-3 Mini, Mistral Small 3)
- Interactive UI with Streamlit
- Dataset upload and analysis
- Code generation from natural language queries
- Safe code execution in sandbox environment
- Performance comparison between different models
- Database storage for conversations and evaluations

## Setup

1. Create a virtual environment and install dependencies:
   \`\`\`
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\\Scripts\\activate
   pip install -r requirements.txt
   \`\`\`

2. Configure your API keys in the .env file:
   - Add your OpenAI API key for GPT-4
   - Add your Anthropic API key for Claude 3.5
   - Configure database settings
   - Set up model paths or endpoints for open-source models

3. Run the application:
   \`\`\`
   streamlit run app.py
   \`\`\`

## Project Structure

- agents/ - LLM agent implementations
- components/ - Core system components
- database/ - Database management
- utils/ - Utility functions and helpers
- evaluation/ - Evaluation metrics and benchmarks
- data/ - Sample datasets and user uploads
- tests/ - Unit tests
- config/ - Configuration files

## Usage

1. Select an AI model in the sidebar
2. Upload your dataset or use a sample dataset
3. Enter your query about the dataset
4. View and run the generated code
5. Compare results between different models

## Team

- Team Member 1: GPT-4/OpenAI model integration
- Team Member 2: UI application in Streamlit and Claude 3.5/3.7 integration
- Team Member 3: Open-source model (Phi-3/Mistral) integration

## License

This project is for academic purposes only.
EOF

# Create a run script
echo -e "\n${YELLOW}Creating run script...${NC}"
cat > run.sh << EOF
#!/bin/bash
# Run the Data Analysis LLM Agent

# Activate virtual environment
source venv/bin/activate

# Run the Streamlit app
streamlit run app.py
EOF

chmod +x run.sh

echo -e "\n${GREEN}Setup complete!${NC}"
echo -e "\n${YELLOW}To run the application:${NC}"
echo -e "1. Edit the .env file with your API keys"
echo -e "2. Run: ${GREEN}./run.sh${NC}"
echo -e "\nHappy data analyzing!"