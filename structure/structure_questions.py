import openai
import json

openai.api_key = "YOUR_API_KEY"

def structure_questions(chunk, prompt_template):
    system_prompt = open(prompt_template, "r").read()
    response = openai.ChatCompletion.create(
        model="gpt-4-turbo",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": chunk}
        ],
        temperature=0.2
    )
    output = response['choices'][0]['message']['content']
    try:
        return json.loads(output)
    except json.JSONDecodeError:
        print("Warning: Failed to parse JSON. Fix manually.")
        return []
