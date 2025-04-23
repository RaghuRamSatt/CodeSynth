import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go

# model_names = ["claude", "gemini", "llama", "o4mini", "openai"]
# categories = ["mean", "matplotlib", "numpy", "pandas", "scipy",
#               "difficult-rewrite", "origin", "semantic", "surface"]

# structure = {
#     model: {
#         key: {"mean": None, "count": None}
#         for key in categories
#     } for model in model_names
# }

# for model in model_names:
#     with open(f"results/{model}-selected-result.txt") as file:
#         for line in file:
#             parts = line.strip().lower().split()
#             if not parts:
#                 continue
#             for key in categories:
#                 if parts[0] == key:
#                     try:
#                         # Assume the format is: key count mean
#                         count = float(parts[1])
#                         mean = float(parts[2])
#                         structure[model][key]["count"] = count
#                         structure[model][key]["mean"] = mean
#                     except (IndexError, ValueError):
#                         pass  # skip malformed lines

# library_keys = ["matplotlib", "numpy", "pandas", "scipy"]
# perturbation_keys = ["difficult-rewrite", "origin", "semantic", "surface"]

# # Convert structure into rows
# rows = []
# for model, metrics in structure.items():
#     for key, values in metrics.items():
#         if key == "mean" or values["mean"] is None:
#             continue  # skip overall mean or missing data

#         category_type = "library" if key in library_keys else "perturbation_type"

#         rows.append({
#             "model": model,
#             "category": key,
#             "category_type": category_type,
#             "mean": values["mean"],
#             "count": values["count"]
#         })

# # Create DataFrame
# df = pd.DataFrame(rows)
# df.to_csv("results/collated_ds1000.csv", header=True, index=False)

df = pd.read_csv("results/collated_ds1000.csv")

model_name_map = {
    'claude': 'Claude3.5-Sonnet',
    'llama': 'Groq-Llama-3.3-70b',
    'openai': 'GPT-4.1',
    'o4mini': 'GPT-o4mini',
    'gemini': 'Gemini'
}

def plot_grouped_bars(df, category_type, title):
    filtered = df[df['category_type'] == category_type]
    categories = filtered['category'].unique()
    models = filtered['model'].unique()

    fig = go.Figure()

    bar_width = 0.2
    x = list(categories)

    for model in models:
        model_data = filtered[filtered['model'] == model]
        y_vals = []
        patterns = []

        for cat in categories:
            val = model_data[model_data['category'] == cat]['mean']
            y = val.values[0] if not val.empty else 0
            y_vals.append(y)

        # Find max per category
        max_vals = filtered.groupby('category')['mean'].max().to_dict()
        patterns = ['/' if y == max_vals[cat] else '' for y, cat in zip(y_vals, categories)]

        fig.add_trace(go.Bar(
            x=x,
            y=y_vals,
            name=model_name_map.get(model, model),
            marker_pattern_shape=patterns,
        ))

    fig.update_layout(
        barmode='group',
        title={'text':f"<b>{title}</b>"},
        xaxis=dict(
            title={'text': f"<b>{category_type}</b>"},
            tickfont=dict(family='Arial Black', size=12, color='black')
        ),
        yaxis=dict(
            title={'text': "<b>Mean Score</b>"},
            tickfont=dict(family='Arial Black', size=12, color='black')
        ),
        legend=dict(
            font=dict(family='Arial Black', size=12, color='black')
        ),
        xaxis_title={'text':f"<b>{category_type}</b>"},
        yaxis_title={'text':f"<b>Mean Score</b>"},
        bargap=0.2
    )

    fig.show()

# Plot for libraries
plot_grouped_bars(df, 'library', 'Model Comparison on Libraries')

# Plot for perturbation types
plot_grouped_bars(df, 'perturbation_type', 'Model Comparison on Perturbation Types')