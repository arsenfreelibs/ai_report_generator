import os
import json
from flask import Flask, request, jsonify
from config import METADATA_PATH, MODEL_PATH, print_config
from js_code_generator import JSCodeGenerator

app = Flask(__name__)

# Initialize the code generator
js_generator = None

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

def start_server(host='0.0.0.0', port=5000, debug=False):
    """Start the API server"""
    print_config()
    print(f"Starting API server on {host}:{port}")
    app.run(host=host, port=port, debug=debug)

if __name__ == "__main__":
    # Get port from environment variable or use default
    port = int(os.environ.get("PORT", 5000))
    start_server(port=port)
