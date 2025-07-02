import json
from openai import OpenAI
import google.generativeai as genai
import anthropic
import config
import os
import requests
from typing import Dict, Optional
from dotenv import load_dotenv
from datetime import datetime
from PIL import Image
import base64
from io import BytesIO

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

    def generate_diagram_openai(self, prompt: str, size: str = "1024x1024", output_dir: str = "generated_diagrams", filename: str = None):
        """
        Generates an image using GPT-1 , downloads it, and saves it to a specified directory.
        Args:
            prompt: The text prompt to send to GPT-1.
            size: The desired size of the image (e.g., "1024x1024").
            output_dir: The directory where the generated image will be saved.
            filename: Custom filename for the image (optional). If not provided, uses "generated_diagram.png".
        Returns:
            The file path of the saved image or None if an error occurred.
        """
        try:
            #print(f"🖼️  Generating GPT-1 image from prompt:\n{prompt}\n")
            client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
            
            response = client.images.generate(
                model="gpt-image-1",
                prompt=prompt,
                background="auto",
                n=1,
                quality="high",
                size="1024x1024",
                output_format="png",
                moderation="auto",
            )
            
            # Print token usage details if available
            if hasattr(response, 'usage') and response.usage:
                usage = response.usage
                print(f"📊 Token Usage Details:")
                print(f"   Total Tokens: {usage.total_tokens}")
                print(f"   Input Tokens: {usage.input_tokens}")
                print(f"   Output Tokens: {usage.output_tokens}")
                
                if hasattr(usage, 'input_tokens_details') and usage.input_tokens_details:
                    details = usage.input_tokens_details
                    print(f"   Input Token Breakdown:")
                    print(f"     Text Tokens: {details.text_tokens}")
                    print(f"     Image Tokens: {details.image_tokens}")
            else:
                print("📊 Token usage information not available in response")
            
            # Get the image data from the response
            b64_data = response.data[0].b64_json
            image_bytes = base64.b64decode(b64_data)

            # Create output directory if it doesn't exist
            os.makedirs(output_dir, exist_ok=True)
            
            # Save the image to the specified output directory
            img = Image.open(BytesIO(image_bytes))
            
            # Use custom filename if provided, otherwise use default
            if filename:
                # Ensure filename has .png extension
                if not filename.endswith('.png'):
                    filename += '.png'
                filepath = os.path.join(output_dir, filename)
            else:
                filepath = os.path.join(output_dir, "generated_diagram.png")
                
            img.save(filepath)
            print(f"✅ Image saved to {filepath}")
            return filepath

        except Exception as e:
            print(f"❌ Error during GPT-1 image generation or download")
            return None

    def generate_question_from_image_openai(self, prompt: str, size: str = "1024x1024", output_dir: str = "generated_diagrams"):
        """
        Generates a question from an image using GPT-1 , downloads it, and saves it to a specified directory.
        """

        client = OpenAI()
        image_url_path="https://i.postimg.cc/jdvtSwQV/input-question.jpg"
        #prompt = "Can you extract the question and image details from AP Physics Question Image?"    
        prompt = "Can you generate similar question and image as the one in the image?"
        response = client.responses.create(
            model="gpt-4.1",
            input=[
                {
                    "role": "user",
                    "content": [
                            {
                            "type": "input_text", "text": prompt
                            },
                            {
                            "type": "input_image", "image_url": image_url_path
                            }
                    ]
                }
            ]
        )
        print('*****Raw response*****', response)
        print('\n*****************************')
        # Extract and print the question and diagram details
        if response.output and len(response.output) > 0:
            output_message = response.output[0]
            if output_message.content and len(output_message.content) > 0:
                extracted_text = output_message.content[0].text
                
                # Parse and separate question and diagram details
                question_text = ""
                diagram_details = ""
                
                # Split by the separator "---"
                parts = extracted_text.split("---")
                
                if len(parts) >= 2:
                    # Extract question part (before "---")
                    question_part = parts[0].strip()
                    if "**Extracted Question:**" in question_part:
                        question_text = question_part.replace("**Extracted Question:**", "").strip()
                    
                    # Extract diagram details part (after "---")
                    diagram_part = parts[1].strip()
                    if "**Image Details:**" in diagram_part:
                        diagram_details = diagram_part.replace("**Image Details:**", "").strip()
                else:
                    # If no separator found, treat entire text as question
                    question_text = extracted_text.strip()
                
                # Print question text separately
                print("=" * 80)
                print("📝 EXTRACTED QUESTION TEXT")
                print("=" * 80)
                print(question_text)
                print()
                
                # Print diagram details separately
                if diagram_details:
                    print("=" * 80)
                    print("🖼️  DIAGRAM DETAILS")
                    print("=" * 80)
                    print(diagram_details)
                    print("=" * 80)
                else:
                    print("=" * 80)
                    print("🖼️  DIAGRAM DETAILS")
                    print("=" * 80)
                    print("No diagram details found in the response.")
                    print("=" * 80)
                
                return extracted_text
            else:
                print("❌ No content found in response")
                return None
        else:
            print("❌ No output found in response")
            return None

    def generate_image_from_image_openai(self, prompt: str, size: str = "1024x1024", output_dir: str = "generated_diagrams"):
        client = OpenAI()
        prompt = """
        Can you generate a similar question and image as the one in the image?
        """

        result = client.images.edit(
            model="gpt-image-1",
            image=[
                open("data/input_questions/input_question.jpg", "rb")  
            ],
            prompt=prompt
        )

        image_base64 = result.data[0].b64_json
        image_bytes = base64.b64decode(image_base64)

        # Save the image to a file
        with open("output_question.png", "wb") as f:
            f.write(image_bytes)
   
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
            model_instance = genai.GenerativeModel(model)
            combined_system_prompt = f"You are {model}. {system_prompt}"
            prompt_content = combined_system_prompt + "\n" + user_prompt
            response = model_instance.generate_content(contents=[prompt_content])
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


def test_image_analysis():
    """
    Test method for generate_question_from_image_openai functionality
    """
    print("=" * 50)
    print("TEST: Image Analysis")
    print("=" * 50)
    
    # Create a simple config for testing
    test_config = {
        "openai_llm_model": "gpt-4o",
        "anthropic_llm_model": "claude-3-5-sonnet-20241022",
        "gemini_llm_model": "gemini-1.5-pro",
        "deepseek_llm_model": "deepseek-chat",
        "grok_llm_model": "grok-3"
    }
    
    # Initialize LLMConnections
    llm_connections = LLMConnections(test_config)
    
    prompt = "Can you extract the question and image details from AP Physics Question Image?"
    size = "1024x1024"
    output_dir = "generated_diagrams"
    
    print("🚀 Starting image analysis...")
    result = llm_connections.generate_question_from_image_openai(
        prompt=prompt,
        size=size,
        output_dir=output_dir
    )
    
    if result:
        print(f"\n🎯 Analysis Result:\n{result}")
    else:
        print("❌ Failed to analyze image")
    
    return result


def test_image_generation_from_image():
    """
    Test method for generate_image_from_image_openai functionality
    """
    print("=" * 50)
    print("TEST: Image Generation from Image")
    print("=" * 50)
    
    # Create a simple config for testing
    test_config = {
        "openai_llm_model": "gpt-4o",
        "anthropic_llm_model": "claude-3-5-sonnet-20241022",
        "gemini_llm_model": "gemini-1.5-pro",
        "deepseek_llm_model": "deepseek-chat",
        "grok_llm_model": "grok-3"
    }
    
    # Initialize LLMConnections
    llm_connections = LLMConnections(test_config)
    
    image_prompt = "Can you generate a similar question and image as the one in the image?"
    image_size = "1024x1024"
    image_output_dir = "generated_images"
    
    print("🎨 Starting image generation from image...")
    try:
        llm_connections.generate_image_from_image_openai(
            prompt=image_prompt,
            size=image_size,
            output_dir=image_output_dir
        )
        print("✅ Image generation completed successfully!")
        print("📁 Check 'output_question.png' for the generated image")
        return True
    except Exception as e:
        print(f"❌ Failed to generate image: {e}")
        print("💡 Make sure 'data/input_questions/input_question.jpg' exists")
        return False


def test_diagram_generation():
    """
    Test method for generate_diagram_openai functionality
    """
    print("=" * 50)
    print("TEST: Diagram Generation")
    print("=" * 50)
    
    # Create a simple config for testing
    test_config = {
        "openai_llm_model": "gpt-4o",
        "anthropic_llm_model": "claude-3-5-sonnet-20241022",
        "gemini_llm_model": "gemini-1.5-pro",
        "deepseek_llm_model": "deepseek-chat",
        "grok_llm_model": "grok-3"
    }
    
    # Initialize LLMConnections
    llm_connections = LLMConnections(test_config)
    
    diagram_prompt = "Create a diagram showing the forces acting on a pendulum"
    diagram_size = "1024x1024"
    diagram_output_dir = "generated_diagrams"
    
    print("📊 Starting diagram generation...")
    try:
        result = llm_connections.generate_diagram_openai(
            prompt=diagram_prompt,
            size=diagram_size,
            output_dir=diagram_output_dir
        )
        if result:
            print("✅ Diagram generation completed successfully!")
            print(f"📁 Diagram saved to: {result}")
        else:
            print("❌ Failed to generate diagram")
        return result
    except Exception as e:
        print(f"❌ Failed to generate diagram: {e}")
        return None


def main():
    """
    Main method to demonstrate the functionality
    """
    print("🧪 LLM Connections Test Suite")
    print("Choose a test to run:")
    print("1. Image Analysis")
    print("2. Image Generation from Image")
    print("3. Diagram Generation")
    print("4. Run All Tests")
    
    # For now, run all tests
    # You can modify this to accept user input or call specific tests
    #print("\n" + "=" * 60)
    #print("RUNNING ALL TESTS")
    print("=" * 60)
    
    # Test 1: Image Analysis
    #test_image_analysis()
    
    # Test 2: Image Generation from Image
    test_image_generation_from_image()
    
    # Test 3: Diagram Generation
    test_diagram_generation()
    
    print("\n" + "=" * 60)
    print("ALL TESTS COMPLETED")
    print("=" * 60)


if __name__ == "__main__":
    main() 