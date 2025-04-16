"""
Prompt Templates - Templates for different LLM agents
"""

# Templates for OpenAI GPT-4
OPENAI_PROMPT_TEMPLATES = {
    "code_generation": """
You are a senior data scientist who specializes in Python programming for data analysis. 
Generate well-documented Python code to solve the following task:

User Query: {user_prompt}

Dataset Information:
{dataset_info}

Generate Python code that addresses the user's query. The code should:
1. Be well-documented with comments explaining the approach
2. Use pandas, numpy, and visualization libraries appropriately
3. Handle potential errors (like missing data)
4. Produce clear visualizations where appropriate
5. Return insightful analysis results

Only provide the Python code, no explanations or conversation.
""",

    "question_answering": """
You are a data science expert. Answer the following question about code or data analysis:

Question: {user_question}

Additional Context:
{context}

Provide a clear, accurate, and educational answer. Use technical terms appropriately but make sure the explanation is accessible.
""",

    "code_improvement": """
You are a Python code optimization expert. You need to improve the following code based on user feedback.

Original Code:
```python
{original_code}
```

User Feedback: {user_feedback}

Provide an improved version of the code that addresses the user's feedback while maintaining the original functionality.
Only provide the improved code, no explanations or conversation.
"""
}

# Templates for Claude 3.5/3.7
CLAUDE_PROMPT_TEMPLATES = {
    "code_generation": """
<task>
Generate Python code to analyze a dataset based on a user query
</task>

<user_query>
{user_prompt}
</user_query>

<dataset_information>
{dataset_info}
</dataset_information>

<instructions>
Create Python code that effectively addresses the user's query. The code should:
1. Use pandas, numpy, and appropriate visualization libraries
2. Include clear comments explaining the approach and key steps
3. Handle potential errors like missing data or invalid types
4. Produce informative visualizations where appropriate
5. Return meaningful insights from the analysis

Provide only the Python code with no additional explanations or conversation.
</instructions>
""",

    "question_answering": """
<task>
Answer a data science question based on context provided
</task>

<question>
{user_question}
</question>

<context>
{context}
</context>

<instructions>
Provide a clear, accurate, and educational answer to the question. Use technical terms appropriately but ensure the explanation is accessible to someone with basic data science knowledge.
</instructions>
""",

    "code_improvement": """
<task>
Improve Python code based on user feedback
</task>

<original_code>
{original_code}
</original_code>

<user_feedback>
{user_feedback}
</user_feedback>

<instructions>
Create an improved version of the code that addresses the user's feedback while maintaining the original functionality. Focus on:
1. Code efficiency and performance
2. Readability and maintainability
3. Best practices for Python and data science
4. Error handling and edge cases

Provide only the improved code with no additional explanations.
</instructions>
"""
}

# Templates for open-source models (more structured and detailed to help these models)
OPENSOURCE_PROMPT_TEMPLATES = {
    "code_generation": """
TASK: Generate Python code for data analysis

USER QUERY:
{user_prompt}

DATASET INFORMATION:
{dataset_info}

INSTRUCTIONS:
You must generate Python code that analyzes the dataset according to the user's query.

The code must:
- Use pandas for data manipulation
- Use numpy for numerical operations
- Use matplotlib or seaborn for visualizations
- Include detailed comments explaining each step
- Handle errors like missing data or invalid types
- Produce clear visualizations where appropriate

FORMAT:
- Return ONLY the Python code, nothing else
- Use ```python and ``` to mark code blocks
- Include appropriate imports at the top

IMPORTANT NOTES:
- The variable 'df' already contains the loaded dataset
- Focus on creating clean, efficient code
- Make sure visualizations have proper titles and labels
- Handle potential errors gracefully
""",

    "question_answering": """
TASK: Answer a data science question

QUESTION:
{user_question}

CONTEXT:
{context}

INSTRUCTIONS:
Provide a clear, accurate, and helpful answer to the question using the context provided.

Your answer should:
- Be technically accurate
- Use appropriate data science terminology
- Explain concepts in an educational manner
- Provide examples where helpful
- Avoid unnecessary jargon

FORMAT:
- Focus on giving the most direct and helpful answer
- Use markdown formatting where appropriate
- Be concise but thorough
""",

    "code_improvement": """
TASK: Improve Python code based on feedback

ORIGINAL CODE:
```python
{original_code}
```

USER FEEDBACK:
{user_feedback}

INSTRUCTIONS:
Create an improved version of the code that addresses the user's feedback.

Your improved code should:
- Maintain the original functionality
- Address all issues mentioned in the feedback
- Be more efficient and readable
- Follow Python best practices
- Have proper error handling
- Use appropriate data science libraries

FORMAT:
- Return ONLY the improved Python code, nothing else
- Use ```python and ``` to mark code blocks
- Include detailed comments explaining your improvements
"""
}