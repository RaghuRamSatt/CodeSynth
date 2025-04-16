# Import required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Calculate the correlation between sepal length and petal length
try:
    correlation = df['sepal length (cm)'].corr(df['petal length (cm)'])
    
    # Print the correlation coefficient
    print(f"Correlation between sepal length and petal length: {correlation:.4f}")
    
    # Create a scatter plot to visualize the relationship
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='sepal length (cm)', y='petal length (cm)', data=df, hue='species', palette='deep')
    plt.title('Scatter Plot: Sepal Length vs Petal Length')
    plt.xlabel('Sepal Length (cm)')
    plt.ylabel('Petal Length (cm)')
    
    # Add a best fit line
    x = df['sepal length (cm)']
    y = df['petal length (cm)']
    m, b, r, p, std_err = stats.linregress(x, y)
    plt.plot(x, m*x + b, color='red', linestyle='--', label=f'Best Fit Line (r={r:.2f})')
    plt.legend()
    
    # Show the plot
    plt.tight_layout()
    plt.show()
    
    # Create a heatmap of the correlation matrix
    plt.figure(figsize=(8, 6))
    correlation_matrix = df.drop('species', axis=1).corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, center=0)
    plt.title('Correlation Matrix of Iris Dataset Features')
    plt.tight_layout()
    plt.show()

except KeyError as e:
    print(f"Error: One or more required columns not found in the dataset. {e}")
except Exception as e:
    print(f"An unexpected error occurred: {e}")