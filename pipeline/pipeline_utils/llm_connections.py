import json
from openai import OpenAI
import google.generativeai as genai
import anthropic
import config
import os
from typing import Dict, Optional
from dotenv import load_dotenv

# Load environment variables
load_dotenv(os.path.join(os.path.dirname(__file__), '.env'))

# API configurations
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
ANTHROPIC_API_KEY = os.getenv('ANTHROPIC_API_KEY')
DEEPSEEK_API_KEY = os.getenv('DEEPSEEK_API_KEY')
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
EMBEDDING_MODEL = os.getenv('EMBEDDING_MODEL', 'text-embedding-3-small')
LLM_MODEL = os.getenv('LLM_MODEL', 'claude-3-opus-20240229')

# Initialize API clients
openai_client = OpenAI(api_key=OPENAI_API_KEY)
anthropic_client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
genai.configure(api_key=GEMINI_API_KEY)

def _call_gemini_api(system_prompt, user_prompt):
    """
    Private method to call the Gemini API.
    """
    try:
        print('Calling Gemini API for content generation...')
        model = genai.GenerativeModel(LLM_MODEL)
        prompt_content = system_prompt + "\n" + user_prompt
        response = model.generate_content(contents=[prompt_content])
        print('Gemini API call successful.')
        ai_response_content = response.text
        print('AI response received.')
        try:
            json.loads(ai_response_content)
            return ai_response_content
        except json.JSONDecodeError:
            print("Warning: Gemini response is not valid JSON. Returning as raw text.")
            return ai_response_content
    except Exception as e:
        print(f"Error during Gemini API call or processing: {e}")
        return None

def _call_deepseek_api(system_prompt, user_prompt):
    """
    Private method to call the DeepSeek API.
    """
    try:
        print('Calling DeepSeek API for content generation...')
        client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url="https://api.deepseek.com")
        completion = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            response_format={"type": "json_object"},
            stream=False,
            timeout=30,
            max_tokens=1000,
        )
        print('DeepSeek API call successful.')
        ai_response_content = completion.choices[0].message.content
        print('AI response received.')
        return ai_response_content
    except Exception as e:
        print(f"Error during DeepSeek API call or processing: {e}")
        return None

def _call_anthropic_api(system_prompt, user_prompt):
    """
    Private method to call the Anthropic Claude API.
    """
    try:
        print('Calling Anthropic Claude API for content generation...')
        system_message = "You are a direct question generator. Do not ask any clarifying questions. Generate questions immediately based on the provided parameters."
        message = anthropic.messages.create(
            model=LLM_MODEL,
            max_tokens=8000,
            system=system_message,
            messages=[
                {
                    "role": "user",
                    "content": f"{system_prompt}\n\n{user_prompt}"
                }
            ],
            temperature=0.6
        )
        print('Anthropic API call successful.')
        ai_response_content = message.content[0].text
        print('AI response received.')
        return ai_response_content
    except Exception as e:
        print(f"Error during Anthropic API call or processing: {e}")
        return None

def _call_openai_api(system_prompt, user_prompt):
    """
    Private method to call the OpenAI API.
    """
    try:
        print('Calling OpenAI API for content generation...')
        client = OpenAI()
        completion = client.chat.completions.create(
                model="gpt-4o-2024-08-06",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.0
            )
        
        print('OpenAI API call successful.')
        ai_response_content = completion.choices[0].message.content
        #print('AI response received.', ai_response_content)
        return ai_response_content
    except Exception as e:
        print(f"Error during OpenAI API call or processing: {e}")
        return None

def call_llm_api(provider: str, system_prompt: str, user_prompt: str) -> Optional[str]:
    """
    Call the appropriate LLM API based on the provider
    Args:
        provider: The LLM provider to use (openai, anthropic, gemini, deepseek)
        system_prompt: The system prompt to use
        user_prompt: The user prompt to use
    Returns:
        The generated content or None if there was an error
    """
    try:
        if provider == "openai":
            return _call_openai_api(system_prompt, user_prompt)
        elif provider == "anthropic":
            return _call_anthropic_api(system_prompt, user_prompt)
        elif provider == "gemini":
            return _call_gemini_api(system_prompt, user_prompt)
        elif provider == "deepseek":
            return _call_deepseek_api(system_prompt, user_prompt)
        else:
            print(f"Unsupported provider: {provider}")
            return None
    except Exception as e:
        print(f"Error calling {provider} API: {e}")
        return None 