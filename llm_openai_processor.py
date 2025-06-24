import openai
from typing import Optional
import os

class LLMOpenAIProcessor:
    """Handles OpenAI API calls for language model processing"""

    def __init__(self, api_key: str, model_name: str = "gpt-3.5-turbo"):
        """Initialize the OpenAI processor"""
        self.api_key = api_key
        self.model_name = model_name
        self.client = None
        
        # Set the API key
        openai.api_key = self.api_key

    def initialize_model(self):
        """Initialize the OpenAI client"""
        print(f"Initializing OpenAI model {self.model_name}...")
        
        try:
            # Initialize OpenAI client
            self.client = openai.OpenAI(api_key=self.api_key)
            
            # Test connection with a simple request
            test_response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": "Hello"}],
                max_tokens=5
            )
            print("OpenAI client initialized successfully")
            return self.client, None  # Return client and None for model (to match interface)
            
        except Exception as e:
            print(f"Error initializing OpenAI client: {str(e)}")
            raise e

    def generate_response(self, prompt: str, max_tokens: int = 4096, temperature: float = 0.2) -> str:
        """Generate a response using OpenAI API"""
        if not self.client:
            self.initialize_model()

        try:
            # Create chat completion
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that generates JavaScript code."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=0.95
            )
            
            # Extract the response content
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            print(f"Error generating response from OpenAI: {str(e)}")
            return f"Error: {str(e)}"
