import os
import pandas as pd
from components.execution_engine import ExecutionEngine

# Initialize execution engine
engine = ExecutionEngine()

# Test with iris.csv
iris_path = "/mnt/d/DS Northeastern/DS 5983 - LLMS/data-analysis-llm-agent/CodeSynth/data/sample_datasets/iris.csv"

# Simple code to test
test_code = """
import pandas as pd
import numpy as np

# Load the dataset
df = pd.read_csv(dataset_path)
print("Dataset shape:", df.shape)
print("First 5 rows:")
print(df.head())
print("Summary statistics:")
print(df.describe())
"""

# Run the test
result = engine.execute_code(test_code, dataset_path=iris_path)

# Print results
print("Success:", result["success"])
print("Output:", result["output"])
if not result["success"]:
    print("Error:", result["error"])