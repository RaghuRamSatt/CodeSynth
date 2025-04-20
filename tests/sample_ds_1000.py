import os
import json
import gzip
import random
import anthropic
from tqdm import tqdm
import numpy as np
 
# Set seed for reproducibility
random.seed(1234)
np.random.seed(1234)
 
# Load the DS-1000 dataset
ds1000 = [json.loads(l) for l in gzip.open("data/ds1000.jsonl.gz", "rt").readlines()]
 
# Filter for the requested libraries
target_libraries = ['Numpy', 'Pandas', 'Matplotlib', 'Scipy']
filtered_problems = [p for p in ds1000 if p['metadata']['library'] in target_libraries]
 
# Group by library to ensure representation
problems_by_library = {}
for p in filtered_problems:
    lib = p['metadata']['library']
    if lib not in problems_by_library:
        problems_by_library[lib] = []
    problems_by_library[lib].append(p)
 
# Select 15 from EACH library (or all if fewer than 15 exist)
selected_problems = []
for lib in target_libraries:
    if lib in problems_by_library:
        lib_problems = problems_by_library[lib]
        n_select = min(15, len(lib_problems))
        selected = random.sample(lib_problems, n_select)
        selected_problems.extend(selected)
        print(f"Selected {n_select} problems from {lib}")
 
# Save the subset in same format
output_dir = "selected_problems_60"
os.makedirs(output_dir, exist_ok=True)
 
# 1. Save as jsonl.gz for compatibility
with gzip.open(f"{output_dir}/selected_ds1000.jsonl.gz", "wt") as f:
    for p in selected_problems:
        f.write(json.dumps(p) + "\n")
 
# 2. Save individual problems for review
for i, p in enumerate(selected_problems):
    problem_id = p['metadata']['problem_id']
    library = p['metadata']['library']
   
    with open(f"{output_dir}/question_{problem_id}_{library}.txt", "w") as f:
        f.write(f"Problem ID: {problem_id}\n")
        f.write(f"Library: {library}\n")
        f.write(f"Perturbation Type: {p['metadata']['perturbation_type']}\n\n")
        f.write("PROMPT:\n")
        f.write(p['prompt'])
 
print(f"Selected problems and saved to {output_dir}/")
print("Library distribution:")
lib_counts = {}
for p in selected_problems:
    lib = p['metadata']['library']
    lib_counts[lib] = lib_counts.get(lib, 0) + 1
for lib, count in lib_counts.items():
    print(f"  {lib}: {count} problems")