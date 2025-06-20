# SL2 JavaScript Code Generator

This tool converts natural language queries into JavaScript code for the SL2 system.

## Running on RunPod

Follow these steps to run the application on RunPod:

1. **Create a RunPod instance**
   - Sign up for RunPod if you haven't already
   - Create a new pod with PyTorch template
   - Select a GPU with at least 16GB VRAM (e.g. RTX 4000 or better)

2. **Upload files to RunPod**
   - Upload all project files to your RunPod instance
   - Make sure `meta_with_field(with option)_50.json` is included

3. **Install dependencies**
   ```bash
   # Make setup script executable
   chmod +x setup_runpod.sh
   
   # Run the setup script
   ./setup_runpod.sh
   ```

4. **Run the application**
   ```bash
   # Run with default settings
   python main.py
   
   # Or specify a custom metadata path
   METADATA_PATH="/path/to/metadata.json" python main.py
   
   # Or use a different model
   MODEL_PATH="Qwen/Qwen-7B-Chat" python main.py
   ```

5. **Using the API server**
   ```bash
   # Start the API server (default model)
   python api_server.py
   
   # Start with a specific model using MODEL_KEY
   MODEL_KEY="codellama" python api_server.py
   MODEL_KEY="mixtral" python api_server.py
   MODEL_KEY="qwen" python api_server.py
   MODEL_KEY="codellama_small" python api_server.py
   
   # Or specify custom port
   PORT=8080 python api_server.py
   
   # Combine model and port configuration
   MODEL_KEY="codellama" PORT=8080 python api_server.py
   ```

   **Available Models:**
   - `codellama`: CodeLlama-13b-Instruct-hf (default, best for code generation)
   - `codellama_small`: CodeLlama-7b-Instruct-hf (smaller, faster)
   - `mixtral`: Mixtral-8x7B-Instruct-v0.1 (general purpose)
   - `qwen`: Qwen-14B-Chat (multilingual support)
   
   Once the server is running, you can make requests to it:
   
   **JavaScript Code Generation:**
   ```bash
   # Example API request locally
   curl -X POST http://localhost:5000/generate \
     -H "Content-Type: application/json" \
     -d '{"query": "Find all active work orders from the last week"}'
   ```

   ```bash
   # Example API request external
   curl -X POST http://69.30.85.116:22102/generate \
     -H "Content-Type: application/json" \
     -d '{"query": "Find all active work orders from the last week"}'
   ```

   **LLM Chat:**
   ```bash
   # Simple chat request
   curl -X POST http://localhost:5000/chat \
     -H "Content-Type: application/json" \
     -d '{"prompt": "Explain how JavaScript async/await works"}'
   
   # Chat with history
   curl -X POST http://localhost:5000/chat \
     -H "Content-Type: application/json" \
     -d '{
       "prompt": "Can you give me an example?",
       "history": [
         {"role": "user", "content": "What is a Promise in JavaScript?"},
         {"role": "assistant", "content": "A Promise is an object representing the eventual completion or failure of an asynchronous operation."}
       ],
       "max_tokens": 1024,
       "temperature": 0.7
     }'

     curl -X POST http://69.30.85.116:22102/chat \
     -H "Content-Type: application/json" \
     -d '{"prompt": "Explain how JavaScript async/await works"}'

     curl -X POST http://69.30.85.116:22102/chat \
     -H "Content-Type: application/json" \
     -d '{
       "prompt": "Can you give me an example?",
       "history": [
         {"role": "user", "content": "What is a Promise in JavaScript?"},
         {"role": "assistant", "content": "A Promise is an object representing the eventual completion or failure of an asynchronous operation."}
       ],
       "max_tokens": 1024,
       "temperature": 0.7
     }'
   ```

   **Prompt Generation:**
   ```bash
   # Generate prompt locally
   curl -X POST http://localhost:5000/generate-prompt \
     -H "Content-Type: application/json" \
     -d '{"query": "Find all active work orders from the last week"}'

   # Generate prompt externally
   curl -X POST http://69.30.85.116:22102/generate-prompt \
     -H "Content-Type: application/json" \
     -d '{"query": "Show me all projects with status completed in the last month"}'
   ```

   The APIs return JSON responses containing:
   - **Generate endpoint**: Generated JavaScript code, validation status, context information
   - **Chat endpoint**: LLM response, conversation parameters, error handling
   - **Generate-prompt endpoint**: Generated prompt with context, no code execution

## Configuration

### Environment Variables

- `MODEL_KEY`: Choose the model to use (default: `codellama_small`)
- `MODEL_PATH`: Direct path to model (overrides MODEL_KEY)
- `METADATA_PATH`: Path to metadata JSON file
- `PORT`: API server port (default: 5000)
- `LLM_PROCESSOR_TYPE`: Use 'local' or 'openai' (default: 'local')
- `OPENAI_API_KEY`: Required when using OpenAI models
- `OPENAI_MODEL_KEY`: OpenAI model to use (gpt-3.5-turbo, gpt-4, etc.)

### Example configurations:
```bash
# Use CodeLlama 13B model
MODEL_KEY="codellama" python api_server.py

# Use OpenAI GPT-4
LLM_PROCESSOR_TYPE="openai" OPENAI_API_KEY="your-key" OPENAI_MODEL_KEY="gpt-4" python api_server.py

# Custom metadata file
METADATA_PATH="/path/to/custom/metadata.json" python api_server.py
```

## API Endpoints

### POST /generate
Generate JavaScript code from natural language.

**Request:**
```json
{
  "query": "Find all active work orders from the last week",
  "max_tokens": 1024,
  "temperature": 0.2
}
```

**Response:**
```json
{
  "status": "success",
  "query": "Find all active work orders from the last week",
  "code": "// Generated JavaScript code...",
  "is_valid": true,
  "context": {
    "models_used": [...],
    "api_methods_used": [...]
  }
}
```

### POST /generate-prompt
Generate a contextual prompt from natural language query without executing code generation.

**Request:**
```json
{
  "query": "Find all active work orders from the last week"
}
```

**Response:**
```json
{
  "status": "success",
  "query": "Find all active work orders from the last week",
  "prompt": "System prompt with available models, fields, and API methods..."
}
```

**Examples:**
```bash
# Generate prompt for code generation
curl -X POST http://localhost:5000/generate-prompt \
  -H "Content-Type: application/json" \
  -d '{"query": "Find all active work orders from the last week"}'

# Generate prompt for a complex query
curl -X POST http://localhost:5000/generate-prompt \
  -H "Content-Type: application/json" \
  -d '{"query": "Create a report showing projects by status with completion dates"}'
```

### POST /chat
Generate LLM response from prompt with optional conversation history.

**Request:**
```json
{
  "prompt": "Explain async/await in JavaScript",
  "history": [
    {"role": "user", "content": "What is a Promise?"},
    {"role": "assistant", "content": "A Promise is..."}
  ],
  "max_tokens": 512,
  "temperature": 0.7
}
```

**Response:**
```json
{
  "status": "success",
  "prompt": "Explain async/await in JavaScript",
  "response": "async/await is a syntax that makes...",
  "history_used": true,
  "parameters": {
    "max_tokens": 512,
    "temperature": 0.7
  }
}
```

### POST /update-metadata
Update metadata index with new or replacement data.

**Request:**
```json
{
  "action": "add",
  "metadata": {
    "data": {
      "model": [
        {
          "id": "new_model_1",
          "alias": "new_model",
          "name": "New Model",
          "field": [
            {
              "id": "field_1",
              "alias": "status",
              "name": "Status",
              "type": "array_string",
              "options": "{\"values\":{\"active\":\"Active\",\"inactive\":\"Inactive\"}}"
            }
          ]
        }
      ]
    }
  }
}
```

**Response:**
```json
{
  "status": "success",
  "action": "add",
  "updated_models": 1,
  "message": "Successfully added 1 models"
}
```

**Actions:**
- `"replace"`: Replace entire metadata with new data
- `"add"`: Add new models or update existing ones by ID/alias

**Examples:**
```bash
# Add new models to existing metadata
curl -X POST http://localhost:5000/update-metadata \
  -H "Content-Type: application/json" \
  -d '{
    "action": "add",
    "metadata": {
      "data": {
        "model": [
          {
            "id": "new_model_1",
            "alias": "new_model",
            "name": "New Model",
            "field": []
          }
        ]
      }
    }
  }'

# Replace entire metadata
curl -X POST http://localhost:5000/update-metadata \
  -H "Content-Type: application/json" \
  -d '{
    "action": "replace",
    "metadata": {
      "data": {
        "model": [...]
      }
    }
  }'
```

### GET /health
Check API health status.

**Response:**
```json
{
  "status": "healthy",
  "model": "path/to/model"
}
```

## Troubleshooting

If you encounter errors:

1. **Memory issues**: Try using a smaller model
   ```bash
   MODEL_KEY="codellama_small" python api_server.py
   ```

2. **Missing dependencies**: Make sure installation was successful
   ```bash
   pip install -r requirements.txt
   ```

3. **CPU-only mode**: If GPU memory is limited
   ```bash
   DEVICE="cpu" python main.py
   ```

4. **API server issues**: Check that Flask is installed and the port isn't in use
   ```bash
   # Install Flask if needed
   pip install flask
   
   # Check if port 5000 is already in use
   netstat -tuln | grep 5000
   ```

5. **Model loading errors**: Ensure you have sufficient disk space and RAM
   ```bash
   # Check available space
   df -h
   
   # Check memory usage
   free -h
   ```