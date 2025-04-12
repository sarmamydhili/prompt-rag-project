import json
import logging
import openai
import anthropic
import google.generativeai as genai
import config
import re

# Initialize logger for this module
logger = logging.getLogger(__name__)

class LLMConnections:
    @staticmethod
    def get_content_from_deepseek(system_prompt, user_prompt):
        """
        Calls the DeepSeek API with the given system and user prompts.

        Args:
            system_prompt (str): The system prompt for the DeepSeek API.
            user_prompt (str): The user prompt for the DeepSeek API.

        Returns:
            str: The raw content of the AI response from the DeepSeek API.
        """
        try:
            # Check if DeepSeek is enabled in config
            if not hasattr(config, 'DEEPSEEK_API_KEY') or not config.DEEPSEEK_API_KEY:
                logger.error('DeepSeek API is not configured')
                return None
                
            client = openai.OpenAI(
                api_key=config.DEEPSEEK_API_KEY,
                base_url="https://api.deepseek.com/v1"
            )
            
            try:
                message = client.chat.completions.create(
                    #model="deepseek-chat",
                    model="deepseek-reasoner",
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    temperature=0.0
                )
                print('******* DEEPSEEK RESPONSE Done')
                return message.choices[0].message.content
                    
            except openai.APIError as e:
                if "402" in str(e):
                    logger.error('DeepSeek API account has insufficient balance')
                else:
                    logger.error(f"Error during DeepSeek API call: {e}")
                return None
                    
        except Exception as e:
            logger.error(f"Error during DeepSeek API call or processing: {e}", exc_info=True)
            return None

    @staticmethod
    def get_content_from_anthropic(system_prompt, user_prompt):
        """
        Calls the Anthropic Claude API with the given system and user prompts.

        Args:
            system_prompt (str): The system prompt for the Claude API.
            user_prompt (str): The user prompt for the Claude API.

        Returns:
            str: The raw content of the AI response from the Claude API.
        """
        try:
            client = anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY)
            
            message = client.messages.create(
                #model="claude-3-5-haiku-20241022",
                model="claude-3-7-sonnet-latest",
                max_tokens=1024,
                system=system_prompt,
                messages=[
                    {
                        "role": "user",
                        "content": user_prompt
                    }
                ]
            )
            
            text = message.content[0].text
            
            # Try to find JSON in the response
            json_match = re.search(r'(\{"answer":\s*"[A-D]"\})', text)
            if json_match:
                print('******* ANTHROPIC RESPONSE Done')
                return text  # Return full response, let the caller handle parsing
            else:
                logger.error('No JSON answer found in Anthropic response')
                return None
                    
        except Exception as e:
            logger.error(f"Error during Anthropic API call or processing: {e}", exc_info=True)
            return None

    @staticmethod
    def get_content_from_openai(system_prompt, user_prompt):
        """
        Generates content from OpenAI based on system and user prompts.

        Parameters:
        - system_prompt (str): The system prompt to guide the AI's behavior.
        - user_prompt (str): The user prompt containing the specific request.

        Returns:
        - str: The raw content of the AI response.
        """
        try:
            # Initialize OpenAI client
            client = openai.OpenAI()

            # Generate AI completion
            completion = client.chat.completions.create(
                model="gpt-4o-2024-08-06",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.0
            )

            return completion.choices[0].message.content
            print('******* OPENAI RESPONSE Done')    
        except Exception as e:
            logger.error(f"Failed to generate AI content: {e}", exc_info=True)
            return None

    @staticmethod
    def get_content_from_gemini(system_prompt, user_prompt):
        """
        Calls the Gemini API with the given system and user prompts.

        Args:
            system_prompt (str): The system prompt for the Gemini API.
            user_prompt (str): The user prompt for the Gemini API.

        Returns:
            str: The raw content of the AI response from the Gemini API.
        """
        try:
            # Check if Gemini is enabled in config
            if not hasattr(config, 'GEMINI_API_KEY') or not config.GEMINI_API_KEY:
                logger.error('Gemini API is not configured')
                return None

            # Configure Gemini API
            genai.configure(api_key=config.GEMINI_API_KEY)

            # Select the Gemini model
            model = genai.GenerativeModel('gemini-2.0-flash')

            # Create a more explicit system prompt for Gemini
            

            # Combine system and user prompts
            prompt_content = system_prompt + "\n\n" + user_prompt

            # Call the Gemini API to generate content
            response = model.generate_content(
                contents=[prompt_content]
            )

            # Extract AI response content
            ai_response_content = response.text

            # Try to find JSON in the response
            json_match = re.search(r'(\{"answer":\s*"[A-D]"\})', ai_response_content)
            if json_match:
                return ai_response_content  # Return full response, let the caller handle parsing
            else:
                # Try to extract just the answer letter if JSON format fails
                letter_match = re.search(r'[A-D]', ai_response_content)
                if letter_match:
                    answer_letter = letter_match.group(0)
                    return f'{{"answer": "{answer_letter}"}}'
                else:
                    logger.error(f'No valid answer found in Gemini response: {ai_response_content}')
                    return None

        except Exception as e:
            logger.error(f"Error during Gemini API call or processing: {e}", exc_info=True)
            return None

# Create a singleton instance
llm_connections = LLMConnections() 