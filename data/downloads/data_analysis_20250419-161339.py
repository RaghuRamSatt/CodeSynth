# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import chardet

# Load the dataset
def load_dataset(dataset_path):
    # Detect encoding
    with open(dataset_path, 'rb') as f:
        result = chardet.detect(f.read())
    encoding = result['encoding']
    
    # Load with detected encoding
    return pd.read_csv(dataset_path, encoding=encoding)

# Load the dataset
# df = load_dataset('/sandbox/data.csv')  # Uncomment if you need to reload the dataset

# Check for missing values
def check_missing_values(df):
    if df.isnull().values.any():
        print("Missing values found:")
        print(df.isnull().sum())
    else:
        print("No missing values found.")

# Check for missing values
check_missing_values(df)

# Create a scatter plot between x and y columns
def create_scatter_plot(df, x_column, y_column):
    try:
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x=x_column, y=y_column, data=df)
        plt.title(f"Scatter Plot of {x_column} vs {y_column}")
        plt.xlabel(x_column)
        plt.ylabel(y_column)
        plt.show()
    except Exception as e:
        print(f"Error creating scatter plot: {e}")

# Create a scatter plot between x and y columns
create_scatter_plot(df, 'x', 'y')