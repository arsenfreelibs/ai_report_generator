import os
import json
from flask import Flask, request, jsonify
from config import METADATA_PATH, MODEL_PATH, print_config
from js_code_generator import JSCodeGenerator
from llm_processor import LLMProcessor

app = Flask(__name__)

# Initialize the code generator and LLM processor
js_generator = None
llm_processor = None

@app.route('/health', methods=['GET'])
def health_check():
    """API endpoint for health check"""
    return jsonify({
        'status': 'healthy',
        'model': MODEL_PATH,
    })

@app.route('/generate', methods=['POST'])
def generate_code():
    """API endpoint to generate JavaScript code from natural language query"""
    global js_generator
    
    # Initialize generator if not already done
    if js_generator is None:
        print("Initializing JS Code Generator...")
        js_generator = JSCodeGenerator(metadata_path=METADATA_PATH, model_path=MODEL_PATH)
        js_generator.initialize()
    
    # Get request data
    request_data = request.json
    
    # Generate code through API
    result = js_generator.api_generate_js_code(request_data)
    
    return jsonify(result)

@app.route('/chat', methods=['POST'])
def generate_llm_answer():
    """API endpoint to generate LLM answer from prompt with optional history"""
    global llm_processor
    
    # Initialize LLM processor if not already done
    if llm_processor is None:
        print("Initializing LLM Processor...")
        llm_processor = LLMProcessor(MODEL_PATH)
        llm_processor.initialize_model()
    
    try:
        # Get request data
        request_data = request.json
        
        # Extract prompt (required)
        prompt = request_data.get('prompt', '')
        if not prompt:
            return jsonify({
                'status': 'error',
                'error': 'No prompt provided'
            }), 400
        
        # Extract optional parameters
        history = request_data.get('history', [])
        max_tokens = request_data.get('max_tokens', 512)
        temperature = request_data.get('temperature', 0.7)
        
        # Build conversation context if history is provided
        if history:
            conversation = ""
            for message in history:
                role = message.get('role', 'user')
                content = message.get('content', '')
                if role == 'user':
                    conversation += f"User: {content}\n"
                elif role == 'assistant':
                    conversation += f"Assistant: {content}\n"
            
            # Add current prompt
            full_prompt = f"{conversation}User: {prompt}\nAssistant:"
        else:
            full_prompt = prompt
        
        # Generate response from LLM
        response = llm_processor.generate_response(
            full_prompt, 
            max_tokens=max_tokens, 
            temperature=temperature
        )
        
        return jsonify({
            'status': 'success',
            'prompt': prompt,
            'response': response,
            'history_used': len(history) > 0,
            'parameters': {
                'max_tokens': max_tokens,
                'temperature': temperature
            }
        })
        
    except Exception as e:
        return jsonify({
            'status': 'error',
            'error': str(e),
            'prompt': request_data.get('prompt', '') if request_data else ''
        }), 500

@app.route('/update-metadata', methods=['POST'])
def update_metadata():
    """API endpoint to update metadata index with new or replacement data"""
    global js_generator
    
    try:
        # Get request data
        request_data = request.json
        
        # Extract metadata update information
        action = request_data.get('action', 'replace')  # 'add' or 'replace'
        metadata_updates = request_data.get('metadata', {})
        
        if not metadata_updates:
            return jsonify({
                'status': 'error',
                'error': 'No metadata provided'
            }), 400
        
        # Initialize generator if not already done
        if js_generator is None:
            print("Initializing JS Code Generator...")
            js_generator = JSCodeGenerator(metadata_path=METADATA_PATH, model_path=MODEL_PATH)
            js_generator.initialize()
        
        # Update metadata through RAG manager
        result = js_generator.rag_manager.update_metadata(metadata_updates, action)
        
        return jsonify({
            'status': 'success',
            'action': action,
            'updated_models': result.get('updated_models', 0),
            'message': result.get('message', 'Metadata updated successfully')
        })
        
    except Exception as e:
        return jsonify({
            'status': 'error',
            'error': str(e),
            'action': request_data.get('action', '') if request_data else ''
        }), 500

@app.route('/generate-prompt', methods=['POST'])
def generate_prompt():
    """API endpoint to generate a prompt from natural language query without full code generation"""
    global js_generator
    
    # Initialize generator if not already done
    if js_generator is None:
        print("Initializing JS Code Generator...")
        js_generator = JSCodeGenerator(metadata_path=METADATA_PATH, model_path=MODEL_PATH)
        js_generator.initialize()
    
    try:
        # Get request data
        request_data = request.json
        
        # Extract query (required)
        query = request_data.get('query', '')
        if not query:
            return jsonify({
                'status': 'error',
                'error': 'No query provided'
            }), 400
        
        prompt = js_generator.generate_server_script_prompt(query)
        
        return jsonify({
            'status': 'success',
            'query': query,
            'prompt': prompt
        })
        
    except Exception as e:
        return jsonify({
            'status': 'error',
            'error': str(e),
            'query': request_data.get('query', '') if request_data else ''
        }), 500

def start_server(host='0.0.0.0', port=5000, debug=False):
    """Start the API server"""
    print_config()
    print(f"Starting API server on {host}:{port}")
    app.run(host=host, port=port, debug=debug)

if __name__ == "__main__":
    # Get port from environment variable or use default
    port = int(os.environ.get("PORT", 5000))
    start_server(port=port)
