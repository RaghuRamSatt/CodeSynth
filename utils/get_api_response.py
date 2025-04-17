def fetch_api_response(ai_model, client, prompt):
    if ai_model == "Anthropic":
        response = client.messages.create(
                        model="claude-3-5-sonnet-20240620",
                        max_tokens=2000,
                        temperature=0.2,
                        messages=[{"role": "user", "content": prompt}]
                    )
        code_text = response.content[0].text

    elif ai_model == "Llama":
        response = client.chat.completions.create(
                        messages=[{"role": "user", "content": prompt}],
                        model="llama-3.3-70b-versatile"
                    )
        code_text = response.choices[0].message.content
    
    elif ai_model == "Open AI":
        response = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="gpt-3.5-turbo"
        )
        code_text = response.choices[0].message.content
    
    else:
        response = client.chat.completions.create(
                        messages=[{"role": "user", "content": prompt}],
                        model="mistral-saba-24b"
                    )
        code_text = response.choices[0].message.content
    
    return code_text