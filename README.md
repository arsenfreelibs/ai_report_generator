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
   # Start the API server
   python api_server.py
   
   # By default, the server runs on port 5000
   # You can change the port with:
   PORT=8080 python api_server.py
   ```
   
   Once the server is running, you can make requests to it:
   ```bash
   # Example API request
   curl -X POST http://localhost:5000/generate \
     -H "Content-Type: application/json" \
     -d '{"query": "Find all active work orders from the last week"}'
   ```
   
   The API returns a JSON response containing:
   - Generated JavaScript code
   - Validation status
   - Context information

## Troubleshooting

If you encounter errors:

1. **Memory issues**: Try using a smaller model
   ```bash
   MODEL_PATH="Qwen/Qwen-7B-Chat" python main.py
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