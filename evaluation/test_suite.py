"""
Test Suite for LLM Model Evaluation
Contains standardized test queries for evaluating model performance
"""

from typing import Dict, List, Any
import os
import json

# Test query categories 
TEST_CATEGORIES = [
    "basic_statistics", 
    "data_cleaning", 
    "correlation_analysis",
    "time_series_analysis", 
    "categorical_analysis",
    "visualization",
    "advanced_analysis"
]

# Standard test queries by category
TEST_QUERIES = {
    "basic_statistics": [
        "Calculate the mean, median, and standard deviation for all numeric columns in the dataset.",
        "What are the key descriptive statistics for this dataset?",
        "Show me a summary of the dataset with the main statistical measures.",
        "Find the min, max, and quartile values for each numeric column."
    ],
    
    "data_cleaning": [
        "Clean the dataset by handling missing values and removing duplicates.",
        "Identify and handle outliers in the numeric columns of this dataset.",
        "Standardize the numeric columns and encode categorical variables for analysis.",
        "Check for and handle inconsistent data formats in the dataset."
    ],
    
    "correlation_analysis": [
        "Find and visualize the correlations between all numeric variables in the dataset.",
        "What variables have the strongest correlation with the target column?",
        "Create a correlation matrix and highlight significant relationships.",
        "Analyze the relationship between variable X and Y, including statistical significance."
    ],
    
    "time_series_analysis": [
        "Plot the trend of the main variable over time and identify seasonality patterns.",
        "Decompose the time series data into trend, seasonal, and residual components.",
        "Create a forecast model for the next 6 periods based on historical data.",
        "Analyze and visualize how the variable has changed over different time periods."
    ],
    
    "categorical_analysis": [
        "Compare the distribution of the target variable across different categories.",
        "Create a breakdown analysis of the main metric by category.",
        "Show the count and percentage distribution for each categorical variable.",
        "Analyze how the main outcome varies for different categorical groups."
    ],
    
    "visualization": [
        "Create an informative dashboard with multiple visualizations of the key metrics.",
        "Generate a visualization that shows the relationship between 3 key variables.",
        "Make a comparative visualization showing the differences between groups.",
        "Create an interactive plot that allows exploring the relationship between variables."
    ],
    
    "advanced_analysis": [
        "Perform a clustering analysis to identify natural groupings in the data.",
        "Build a simple predictive model for the target variable and evaluate its performance.",
        "Conduct feature importance analysis to identify the most influential variables.",
        "Perform dimension reduction using PCA and visualize the results."
    ]
}

# Test datasets to use with the queries (matched to query categories)
TEST_DATASETS = {
    "basic_statistics": ["iris.csv", "diamonds.csv", "tips.csv"],
    "data_cleaning": ["superstore.csv", "sp500.csv"],
    "correlation_analysis": ["iris.csv", "diamonds.csv"],
    "time_series_analysis": ["superstore.csv"],
    "categorical_analysis": ["diamonds.csv", "tips.csv", "iris.csv"],
    "visualization": ["iris.csv", "diamonds.csv", "tips.csv"],
    "advanced_analysis": ["iris.csv", "diamonds.csv", "superstore.csv"]
}

class TestSuite:
    """
    Manages the test suite for evaluating LLM model performance.
    """
    
    def __init__(self, data_dir=None):
        """
        Initialize the test suite.
        
        Args:
            data_dir: Directory containing sample datasets
        """
        if data_dir is None:
        # Hardcoded absolute path to match specific directory structure --- getting error otherwise
            self.data_dir = "/mnt/d/DS Northeastern/DS 5983 - LLMS/data-analysis-llm-agent/CodeSynth/data/sample_datasets"
        else:
            self.data_dir = data_dir
    
        self.test_categories = TEST_CATEGORIES
        self.test_queries = TEST_QUERIES
        self.test_datasets = TEST_DATASETS
        
    def get_test_cases(self, categories=None, complexity=None, limit=None) -> List[Dict[str, Any]]:
        """
        Get test cases for evaluation.
        
        Args:
            categories: List of categories to include (defaults to all)
            complexity: Complexity level ('simple', 'medium', 'complex')
            limit: Maximum number of test cases to return
            
        Returns:
            List of test case dictionaries
        """
        if categories is None:
            categories = self.test_categories
        
        test_cases = []
        
        for category in categories:
            if category not in self.test_queries:
                continue
                
            # Get queries for this category
            queries = self.test_queries[category]
            
            # Get datasets for this category
            datasets = self.test_datasets.get(category, ["iris.csv"])
            
            # Create a test case for each query-dataset combination
            for query in queries:
                for dataset in datasets:
                    test_case = {
                        "category": category,
                        "query": query,
                        "dataset": dataset,
                        "dataset_path": os.path.join(self.data_dir, dataset)
                    }
                    test_cases.append(test_case)
                    
        # Apply limit if specified
        if limit and len(test_cases) > limit:
            return test_cases[:limit]
            
        return test_cases
        
    def save_test_suite(self, filepath="evaluation/test_suite.json"):
        """
        Save the test suite to a JSON file.
        
        Args:
            filepath: Path to save the JSON file
        """
        test_suite_data = {
            "categories": self.test_categories,
            "queries": self.test_queries,
            "datasets": self.test_datasets
        }
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        with open(filepath, 'w') as f:
            json.dump(test_suite_data, f, indent=4)
            
    def load_test_suite(self, filepath="evaluation/test_suite.json"):
        """
        Load the test suite from a JSON file.
        
        Args:
            filepath: Path to the JSON file
        """
        try:
            with open(filepath, 'r') as f:
                test_suite_data = json.load(f)
                
            self.test_categories = test_suite_data.get("categories", TEST_CATEGORIES)
            self.test_queries = test_suite_data.get("queries", TEST_QUERIES)
            self.test_datasets = test_suite_data.get("datasets", TEST_DATASETS)
            
            return True
        except Exception as e:
            print(f"Error loading test suite: {e}")
            return False