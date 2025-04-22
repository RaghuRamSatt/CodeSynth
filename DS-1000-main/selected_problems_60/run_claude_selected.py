
import os
import json
import gzip
import anthropic
from tqdm import tqdm

# Load the selected problems
selected_problems = [json.loads(l) for l in gzip.open("selected_problems_60/selected_ds1000.jsonl.gz", "rt").readlines()]

# Initialize Claude client
client = anthropic.Anthropic(api_key="your_api_key_here")

# Run Claude on each question
ds1000_responses = []
for p in tqdm(selected_problems):
    id = int(p['metadata']['problem_id'])
    prompt = p['prompt']
    
    # Call Claude API
    response = client.messages.create(
        model="claude-3-5-sonnet-20240620",
        system="Write a short code following the given format and indentation. Only provide the exact code completion needed. Don't repeat the context code or add explanations.",
        max_tokens=1024,
        temperature=0,
        messages=[{"role": "user", "content": prompt}],
        stop_sequences=["</code>", "# SOLUTION END", "END SOLUTION"]
    )
    
    code = response.content[0].text
    
    # Clean up the response
    code = code.replace("```python", "").replace("```", "").strip()
    
    ds1000_responses.append({
        'id': id, 
        'code': code, 
        'metadata': p['metadata']
    })

# Save to a JSONL file
with open('selected_problems_60/claude-selected-answers.jsonl', 'w') as f:
    for r in ds1000_responses:
        f.write(json.dumps(r) + '\n')

print("Generated answers for selected problems")
