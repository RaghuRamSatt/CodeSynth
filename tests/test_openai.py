import os
import json
import gzip
import openai
from tqdm import tqdm
from dotenv import dotenv_values

# Config file
config = dotenv_values(".env")

# Load the selected problems
selected_problems = [json.loads(l) for l in gzip.open("selected_problems_60/selected_ds1000.jsonl.gz", "rt").readlines()]
 
# Initialize Claude client
client = openai.OpenAI(api_key=config["OPENAI_API_KEY"])
 
# Run Claude on each question
ds1000_responses = []
for p in tqdm(selected_problems):
    id = int(p['metadata']['problem_id'])
    prompt = p['prompt']
   
    # Call Open AI API
    response = client.chat.completions.create(
                        messages=[
                            {"role": "system", "content": "Write a short code following the given format and indentation. Only provide the exact code completion needed. Don't repeat the context code or add explanations."},
                            {"role": "user", "content": prompt}],
                        model="gpt-4.1",
                        stop=["</code>", "# SOLUTION END", "END SOLUTION"],
                        temperature=0,
                        max_tokens=1024
                    )
    code = response.choices[0].message.content  

    # response = client.messages.create(
    #     model="llama-3.3-70b-versatile",
    #     system="Write a short code following the given format and indentation. Only provide the exact code completion needed. Don't repeat the context code or add explanations.",
    #     max_tokens=1024,
    #     temperature=0,
    #     messages=[{"role": "user", "content": prompt}],
    #     stop_sequences=["</code>", "# SOLUTION END", "END SOLUTION"]
    # )
   
    # code = response.content[0].text
   
    # Clean up the response
    code = code.replace("```python", "").replace("```", "").strip()
   
    ds1000_responses.append({
        'id': id,
        'code': code,
        'metadata': p['metadata']
    })
 
# Save to a JSONL file
with open('selected_problems_60/openai-selected-answers.jsonl', 'w') as f:
    for r in ds1000_responses:
        f.write(json.dumps(r) + '\n')
 
print("Generated answers for selected problems")