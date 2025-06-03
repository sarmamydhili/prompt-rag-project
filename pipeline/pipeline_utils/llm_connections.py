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
#EMBEDDING_MODEL = os.getenv('EMBEDDING_MODEL', 'text-embedding-3-small')
#LLM_MODEL = os.getenv('LLM_MODEL', 'claude-3-opus-20240229')

# Initialize API clients
# openai_client = OpenAI(api_key=OPENAI_API_KEY)
# anthropic_client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
# genai.configure(api_key=GEMINI_API_KEY)

class LLMConnections:
    def __init__(self, config):
        self.task_config = config

    def call_llm_api(self, provider: str, system_prompt: str, user_prompt: str, model: Optional[str] = None, temperature: Optional[float] = None) -> Optional[str]:
        """
        Call the appropriate LLM API based on the provider
        Args:
            provider: The LLM provider to use (openai, anthropic, gemini, deepseek)
            system_prompt: The system prompt to use
            user_prompt: The user prompt to use
            model: The model to use for the API call
            temperature: The temperature to use for the API call
        Returns:
            The generated content or None if there was an error
        """
        # Dynamically select the model based on the provider
        selected_model = model or self.task_config.get(f"{provider}_llm_model")

        try:
            if provider == "openai":
                return self._call_openai_api(system_prompt, user_prompt, model=selected_model, temperature=temperature)
            elif provider == "anthropic":
                return self._call_anthropic_api(system_prompt, user_prompt, model=selected_model, temperature=temperature)
            elif provider == "gemini":
                return self._call_gemini_api(system_prompt, user_prompt, model=selected_model, temperature=temperature)
            elif provider == "deepseek":
                return self._call_deepseek_api(system_prompt, user_prompt, model=selected_model, temperature=temperature)
            elif provider == "grok":
                return self._call_grok_api(system_prompt, user_prompt, model=selected_model, temperature=temperature)
            else:
                print(f"Unsupported provider: {provider}")
                return None
        except Exception as e:
            print(f"Error calling {provider} API: {e}")
            return None

    def _call_openai_api(self, system_prompt, user_prompt, model=None, temperature=0.0):
        """
        Private method to call the OpenAI API.
        """
        try:
            print('Calling OpenAI API for content generation...')
            client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
            completion = client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    temperature=temperature
                )
            
            print('OpenAI API call successful.')
            ai_response_content = completion.choices[0].message.content
            return ai_response_content
        except Exception as e:
            print(f"Error during OpenAI API call or processing: {e}")
            return None

    def _call_gemini_api(self, system_prompt, user_prompt, model=None, temperature=0.0):
        """
        Private method to call the Gemini API.
        """
        try:
            print('Calling Gemini API for content generation...')
            genai.configure(api_key=GEMINI_API_KEY)
            model = genai.GenerativeModel(model)
            prompt_content = system_prompt + "\n" + user_prompt
            response = model.generate_content(contents=[prompt_content])
            print('Gemini API call successful.')
            ai_response_content = response.text
            print('gemini AI response received.')
            return ai_response_content
            #try:
            #    json.loads(ai_response_content)
            #    return ai_response_content
            #except json.JSONDecodeError:
            #    print("Warning: Gemini response is not valid JSON. Returning as raw text.")
            #    return ai_response_content
        except Exception as e:
            print(f"Error during Gemini API call or processing: {e}")
            return None

    def _call_deepseek_api(self, system_prompt, user_prompt, model=None, temperature=0.0):
        """
        Private method to call the DeepSeek API.
        """
        try:
            print('Calling DeepSeek API for content generation...')
            client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url="https://api.deepseek.com")
            completion = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                #response_format={"type": "json_object"},
                stream=False,
                timeout=30,
                max_tokens=1000,
                temperature=temperature
            )
            print('DeepSeek API call successful.')
            ai_response_content = completion.choices[0].message.content
            print('DeepSeek AI response received.')
            return ai_response_content
        except Exception as e:
            print(f"Error during DeepSeek API call or processing: {e}")
            return None

    def _call_anthropic_api(self, system_prompt, user_prompt, model=None, temperature=0.0):
        """
        Private method to call the Anthropic Claude API.
        """
        try:
            print('Calling Anthropic Claude API for content generation...')
            anthropic_client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
            combined_system_prompt = f"You are {model}. {system_prompt}"
            message = anthropic_client.messages.create(
                model=model,
                max_tokens=8000,
                system=combined_system_prompt,
                messages=[
                    {
                        "role": "user",
                        "content": user_prompt
                    }
                ],
                temperature=temperature
            )
            print('Anthropic API call successful.')
            ai_response_content = message.content[0].text
            print('Anthropic AI response received.')
            return ai_response_content
        except Exception as e:
            print(f"Error during Anthropic API call or processing: {e}")
            return None

    def _call_grok_api(self, system_prompt, user_prompt, model='grok-3', temperature=0.0):
        """
        Private method to call the Grok API.
        """
        try:
            print('Calling Grok API for content generation...')
            client = OpenAI(
                api_key=os.getenv('XAI_API_KEY'),  # Ensure this environment variable is set
                base_url="https://api.x.ai/v1"
            )
            completion = client.beta.chat.completions.parse(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=temperature
            )
            print('Grok API call successful.')
            ai_response_content = completion.choices[0].message.content
            print('Grok AI response received.')
            return ai_response_content
        except Exception as e:
            print(f"Error during Grok API call or processing: {e}")
            return None

# Usage
# llm_connections = LLMConnections(config)
# response = llm_connections.call_llm_api(provider='openai', system_prompt='...', user_prompt='...') 