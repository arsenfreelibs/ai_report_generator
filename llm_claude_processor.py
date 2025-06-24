import anthropic
from typing import Optional, List, Dict
import os

class LLMClaudeProcessor:
    """Handles Claude API calls for language model processing"""

    def __init__(self, api_key: str, model_name: str = "claude-3-5-sonnet-20241022"):
        """Initialize the Claude processor"""
        self.api_key = api_key
        self.model_name = model_name
        self.client = None
        
    def initialize_model(self):
        """Initialize the Claude client"""
        print(f"Initializing Claude model {self.model_name}...")
        
        try:
            # Initialize Claude client
            self.client = anthropic.Anthropic(api_key=self.api_key)
            
            # Test connection with a simple request
            test_response = self.client.messages.create(
                model=self.model_name,
                max_tokens=10,
                messages=[{"role": "user", "content": "Hello"}]
            )
            print("Claude client initialized successfully")
            return self.client, None  # Return client and None for model (to match interface)
            
        except Exception as e:
            print(f"Error initializing Claude client: {str(e)}")
            raise e

    def generate_response(self, prompt: str, max_tokens: int = 512, temperature: float = 0.2) -> str:
        """Generate a response using Claude API"""
        if not self.client:
            self.initialize_model()

        try:
            # Create message completion
            response = self.client.messages.create(
                model=self.model_name,
                max_tokens=max_tokens,
                temperature=temperature,
                system="You are a helpful assistant that generates JavaScript code.",
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            
            # Extract the response content
            return response.content[0].text.strip()
            
        except Exception as e:
            print(f"Error generating response from Claude: {str(e)}")
            return f"Error: {str(e)}"

    def generate_chat_response(self, messages: List[Dict], max_tokens: int = 512, temperature: float = 0.7) -> str:
        """Generate a chat response using Claude API with conversation history"""
        if not self.client:
            self.initialize_model()

        try:
            # Convert messages to Claude format if needed
            claude_messages = []
            system_content = None
            
            for msg in messages:
                if msg["role"] == "system":
                    system_content = msg["content"]
                else:
                    claude_messages.append({
                        "role": msg["role"],
                        "content": msg["content"]
                    })
            
            # Create message completion with optional system message
            response_params = {
                "model": self.model_name,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "messages": claude_messages
            }
            
            if system_content:
                response_params["system"] = system_content
            
            response = self.client.messages.create(**response_params)
            
            # Extract the response content
            return response.content[0].text.strip()
            
        except Exception as e:
            print(f"Error generating chat response from Claude: {str(e)}")
            return f"Error: {str(e)}"
