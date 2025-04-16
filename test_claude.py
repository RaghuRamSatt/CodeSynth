import os
import anthropic
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize Anthropic client
api_key = os.getenv("ANTHROPIC_API_KEY")
client = anthropic.Anthropic(api_key=api_key)

# Test a simple request
print("Sending request to Claude...")
response = client.messages.create(
    model="claude-3-5-sonnet-20240620",
    max_tokens=500,
    messages=[
        {"role": "user", "content": "Generate a simple Python script that creates a histogram with some random data."}
    ]
)

# Print response
print("\nResponse received:")
print(response.content[0].text)