import re
import json
from typing import Dict, List
from rag_manager import RagManager
from llm_processor import LLMProcessor

class JSCodeGenerator:
    """Main class for converting natural language to JavaScript code"""

    def __init__(self, metadata_path: str, model_path: str):
        """Initialize the JavaScript code generator"""
        self.rag_manager = RagManager(metadata_path)
        self.llm_processor = LLMProcessor(model_path)

    def initialize(self):
        """Initialize all components"""
        self.rag_manager.initialize()
        self.llm_processor.initialize_model()
        print("JavaScript Code Generator initialized")

    def _create_js_prompt(self, query: str, context: List[Dict]) -> str:
        """Create a prompt for JS code generation with retrieved context"""
        # Categorize context items
        models_info = [item for item in context if item['type'] == 'model']
        report_examples = [item for item in context if item['type'] == 'report_example']
        api_info = [item for item in context if item['type'] == 'js_api']
        
        # Format model information with emphasis on aliases
        models_str = "\n".join([
            f"- Model: {item.get('name', '')} (alias: '{item.get('alias', '')}', id: {item.get('id', '')})"
            for item in models_info
        ])

        fields_info = []
        
        # Extract fields from models with special handling for array_string fields
        for model in models_info:
            model_fields = model.get('fields', [])
            for field in model_fields:
                field_info = {
                    'type': 'field',
                    'name': field.get('name', ''),
                    'alias': field.get('alias', ''),
                    'field_type': field.get('type', ''),
                    'model_name': model.get('name', ''),
                    'model_alias': model.get('alias', ''),
                    'model_id': model.get('id', '')
                }
                
                # Add field values for array_string fields with detailed examples
                if field.get('type') == 'array_string':
                    values = None
                    
                    # Try to get values directly from field
                    if 'values' in field:
                        values = field.get('values', {})
                    
                    # Try to extract from parsed_options
                    elif 'parsed_options' in field:
                        options = field.get('parsed_options', {})
                        if 'values' in options:
                            values = options.get('values', {})
                    
                    # Try to parse options string if it exists
                    elif isinstance(field.get('options'), str):
                        try:
                            options_dict = json.loads(field.get('options', '{}'))
                            if 'values' in options_dict:
                                values = options_dict.get('values', {})
                        except json.JSONDecodeError:
                            pass
                    
                    if values:
                        field_info['values'] = values
                        # Add example query for this specific field's values
                        field_info['example_query'] = f"model.find({{'{field.get('alias')}': '{list(values.keys())[0]}'}}) // Use '{list(values.keys())[0]}', not '{list(values.values())[0]}'"
                
                fields_info.append(field_info)

        # Format report examples with emphasis on client/server script patterns
        report_examples_str = ""
        if report_examples:
            report_examples_str = "\n## Similar Report Examples:\n"
            for example in report_examples:
                report_examples_str += f"\n### Example: {example.get('name', 'Unknown')}\n"
                report_examples_str += f"User Request: {example.get('user_request', '')[:200]}...\n"
                
                # Add relevant code snippets from client and server scripts
                client_script = example.get('client_script', '')
                server_script = example.get('server_script', '')
                
                if server_script:
                    # Extract key patterns from server script
                    server_lines = server_script.split('\n')[:20]  # First 20 lines
                    report_examples_str += f"Server Script Pattern:\n```javascript\n"
                    report_examples_str += "\n".join(server_lines)
                    report_examples_str += "\n```\n"
                
                if client_script:
                    # Extract key patterns from client script
                    client_lines = client_script.split('\n')[:15]  # First 15 lines
                    report_examples_str += f"Client Script Pattern:\n```javascript\n"
                    report_examples_str += "\n".join(client_lines)
                    report_examples_str += "\n```\n"

        # Format field information, grouped by model, with emphasis on aliases and including possible values
        fields_by_model = {}
        array_string_examples = []  # Store example queries for array_string fields
        
        for field in fields_info:
            model_name = field.get('model_name', '')
            model_alias = field.get('model_alias', '')
            key = f"{model_name} (alias: '{model_alias}')"
            if key not in fields_by_model:
                fields_by_model[key] = []
            fields_by_model[key].append(field)
            
            # Collect examples for array_string fields
            if field.get('field_type') == 'array_string' and 'example_query' in field:
                array_string_examples.append({
                    'model_alias': model_alias,
                    'field_alias': field.get('alias'),
                    'example': field.get('example_query')
                })

        fields_str = ""
        for model_info, fields in fields_by_model.items():
            fields_str += f"\nFields for model {model_info}:\n"
            for field in fields:
                field_str = f"- {field.get('name', '')} (alias: '{field.get('alias', '')}', type: {field.get('field_type', '')})"
                
                # Add values information for array_string fields with clear instructions
                if field.get('field_type') == 'array_string' and 'values' in field:
                    values = field.get('values', {})
                    value_pairs = [f"'{k}': '{v}'" for k, v in values.items()]
                    field_str += f"\n  Possible values: {{{', '.join(value_pairs)}}}"
                    field_str += f"\n  IMPORTANT: When filtering, use the key ({list(values.keys())[0:3]}) not the display value"
                
                fields_str += field_str + "\n"
            fields_str += "\n"

        # Format API methods
        api_str = "\n".join([
            f"- {item.get('name', '')}: {item.get('description', '')}\n  Syntax: {item.get('syntax', '')}\n  Example: {item.get('example', '')}"
            for item in api_info
        ])

        # Add specific array_string examples if available
        array_string_examples_str = ""
        if array_string_examples:
            array_string_examples_str = "\n## Critical Examples for array_string Fields:\n"
            for example in array_string_examples:
                array_string_examples_str += f"- For model '{example['model_alias']}', field '{example['field_alias']}': {example['example']}\n"

        # JavaScript code examples for common operations
        js_examples = """
JavaScript Code Examples for SL2 System:

1. Getting a model and finding records (returns array of records):
```javascript
// Get model by its alias (not display name)
const model = await p.getModel('test_db_1'); // Use 'test_db_1', not 'Test DB 1'

// Find records with a filter - returns array of records
// Use field aliases in queries, not display names
const records = await model.find({status: 'active'}); // Use 'status', not 'Status'
return records; // Return the array of records

// For array_string fields, always use the KEY value, not the display value:
// If options are {"values":{"New":"New","Open":"Open"}} - use 'New', not the display value
const results = await model.find({status: 'New'});  // CORRECT - Using the key value
// NOT: const results = await model.find({status: 'New Status'}); // WRONG - Using display value

// Find with multiple conditions - returns array of records
const results = await model.find({
  field_alias1: 'value1', // Always use field aliases
  field_alias2: {'>': 100}
}).order({created_at: 'desc'});
return results; // Return the array of records
```

2. Finding records from the past week:
```javascript
// Get records from the past week
const today = new Date();
const lastWeek = new Date(today);
lastWeek.setDate(today.getDate() - 7);

const model = await p.getModel('model_alias');
const records = await model.find({
  created_at: {'>=': lastWeek.toISOString()}
});
return records;
```

3. Limiting number of records returned:
```javascript
// Get only first N records
const model = await p.getModel('model_alias');
const records = await model.find({status: 'active'}).limit(15);
return records;
```

4. Advanced querying with fetchRecords (returns array of records):
```javascript
// Get records using fetchRecords - extract and return the array from the result
// Always use the model alias
const result = await p.uiUtils.fetchRecords('test_db_1', {
  filter: 'status = "active" AND created_at > "2023-01-01"', // Use field aliases
  fields: {id: 'id', name: 'name'},
  page: {size: 15} // Limit to 15 records
});
return result.data; // Return the array of records
```

5. Processing records before returning:
```javascript
// Query, process, and return records
const records = await model.find({status: 'active'});
// Process records if needed
const processedRecords = records.map(record => {
  return {
    id: record.id,
    status: record.status, // Use field aliases when referencing fields
    // Add any other fields or transformations
  };
});
return processedRecords; // Return the processed array of records
```

6. Error handling pattern:
```javascript
try {
  // Always use model aliases
  const model = await p.getModel('work_order');
  // Always use field aliases in queries
  const records = await model.find({active: true});
  return records; // Return the array of records
} catch (error) {
  console.error('Error:', error);
  return []; // Return empty array in case of error
}
```
"""

        # Construct the system prompt with emphasis on generating both client and server scripts
        system_prompt = """You are an expert JavaScript code generator for the SL2 system. Your task is to convert natural language requests into both CLIENT and SERVER JavaScript code that uses SL2's API.

IMPORTANT:
- You must generate BOTH a client script and a server script
- The server script should handle data retrieval and processing using SL2 API
- The client script should handle UI interactions and display logic
- ALWAYS use model and field aliases in your code, NOT their display names
- For array_string fields, you MUST use the KEY values in the filter (e.g., 'New', 'Open'), NOT the display values
- Follow the patterns from the provided report examples

Server Script Guidelines:
1. Use SL2 API methods like p.getModel(), model.find(), p.uiUtils.fetchRecords()
2. Handle data retrieval, filtering, and processing
3. Return arrays of records or processed data
4. Include error handling with try/catch blocks
5. Use async/await for API calls

Client Script Guidelines:
1. Handle UI interactions and display logic
2. Process data for presentation
3. Manage user interface elements
4. Handle events and user interactions

Follow these guidelines:
1. Only use the provided models, fields, and API methods in your code
2. Always refer to models and fields by their aliases
3. When querying array_string fields, use EXACTLY the key values provided
4. Use the provided report examples as reference patterns
5. Write clean, well-commented code with proper error handling
"""

        # Construct the full prompt
        prompt = f"""
{system_prompt}

## Available Models:
{models_str}

{fields_str}

{array_string_examples_str}

## Available JS API Methods:
{api_str}

{report_examples_str}

{js_examples}

## User Request:
{query}

## Task:
Generate BOTH client and server JavaScript code that fulfills the user's request using the available models, fields, and API methods.

CRITICAL REQUIREMENTS:
1. Generate separate CLIENT and SERVER scripts
2. ALWAYS use model aliases (e.g., 'test_db_1') instead of display names ('Test DB 1')
3. ALWAYS use field aliases (e.g., 'status') instead of display names ('Status')
4. For array_string fields, ALWAYS use the EXACT key values (e.g., 'New', 'Open'), NOT the display values
5. Follow the patterns from the provided report examples
6. Server script should return data, client script should handle UI

## Response Format:
CLIENT_SCRIPT:
```javascript
// Client script code here
```

SERVER_SCRIPT:
```javascript
// Server script code here
```
"""
        return prompt

    def _extract_client_server_scripts(self, response_text: str) -> Dict[str, str]:
        """Extract client and server JavaScript code from model response text"""
        import re
        
        # Initialize default scripts
        client_script = ""
        server_script = ""
        
        # Look for CLIENT_SCRIPT section
        client_pattern = r"CLIENT_SCRIPT:\s*```(?:javascript|js)?\s*([\s\S]*?)```"
        client_match = re.search(client_pattern, response_text, re.IGNORECASE)
        if client_match:
            client_script = client_match.group(1).strip()
        
        # Look for SERVER_SCRIPT section
        server_pattern = r"SERVER_SCRIPT:\s*```(?:javascript|js)?\s*([\s\S]*?)```"
        server_match = re.search(server_pattern, response_text, re.IGNORECASE)
        if server_match:
            server_script = server_match.group(1).strip()
        
        # If no specific sections found, try to extract any JavaScript code blocks
        if not client_script and not server_script:
            js_pattern = r"```(?:javascript|js)?\s*([\s\S]*?)```"
            matches = re.findall(js_pattern, response_text)
            if matches:
                # If only one code block, use it as server script
                if len(matches) == 1:
                    server_script = matches[0].strip()
                # If multiple code blocks, first as client, second as server
                elif len(matches) >= 2:
                    client_script = matches[0].strip()
                    server_script = matches[1].strip()
        
        return {
            'client_script': client_script,
            'server_script': server_script
        }

    def generate_prompt(self, query: str) -> str:
        """Process a natural language query and return JavaScript code"""
        # Ensure query emphasizes returning records if not already mentioned
        enhanced_query = query
        if "return" not in query.lower() and "array" not in query.lower():
            enhanced_query = f"{query} and return the array of records"

        # Retrieve relevant context
        context = self.rag_manager.hybrid_search(enhanced_query, k=15)
        print(f"Retrieved {len(context)} context items for query: '{enhanced_query}'")

        # Create JS generation prompt
        prompt = self._create_js_prompt(enhanced_query, context)
        print(f"prompt: {prompt}")   
        return prompt 

    def generate_js_code(self, query: str) -> Dict:
        """Process a natural language query and return JavaScript code"""
        # Ensure query emphasizes returning records if not already mentioned
        enhanced_query = query
        if "return" not in query.lower() and "array" not in query.lower():
            enhanced_query = f"{query} and return the array of records"

        # Retrieve relevant context using RagManager
        context = self.rag_manager.hybrid_search(enhanced_query, k=25)
        print(f"Retrieved {len(context)} context items for query: '{enhanced_query}'")

        # Create JS generation prompt
        prompt = self._create_js_prompt(enhanced_query, context)
        print(f"prompt: {prompt}")
        
        # Generate response from LLM
        response_text = self.llm_processor.generate_response(prompt, max_tokens=1024, temperature=0.2)

        # Extract client and server scripts from response
        scripts = self._extract_client_server_scripts(response_text)
        
        # Ensure server script returns array of records if it doesn't already
        if scripts['server_script']:
            scripts['server_script'] = self._ensure_code_returns_array(scripts['server_script'])
        
        # Return result
        return {
            "natural_language_query": query,
            "client_script": scripts['client_script'],
            "server_script": scripts['server_script'],
            "context_used": {
                "models": [item for item in context if item['type'] == 'model'][:3],
                "fields": [item for item in context if item['type'] == 'field'][:5],
                "api_methods": [item for item in context if item['type'] == 'js_api'][:5],
            }
        }

    def validate_js(self, js_code: str) -> bool:
        """Basic validation of JavaScript code syntax"""
        # Check for common JS patterns that indicate well-formed code
        has_await = "await" in js_code
        has_const_or_let = re.search(r"(const|let|var)\s+\w+", js_code) is not None
        has_function = "function" in js_code or "=>" in js_code
        has_api_call = "p." in js_code or "model." in js_code
        has_return = "return " in js_code

        # Simple validation - needs at least some of these elements
        basic_validation = has_api_call and (has_const_or_let or has_function or has_await)

        # Check if code likely returns an array of records
        returns_array = False
        if has_return:
            # Check common patterns for returning records
            return_patterns = [
                r"return\s+\w+;",  # return records;
                r"return\s+\w+\.data;",  # return result.data;
                r"return\s+\w+\.map\(",  # return records.map(
                r"return\s+\[",  # return [ (array literal)
                r"return\s+results",  # return results
                r"return\s+filtered",  # return filtered
                r"return\s+matched"  # return matched
            ]

            for pattern in return_patterns:
                if re.search(pattern, js_code):
                    returns_array = True
                    break

        return basic_validation and returns_array


    def interactive_mode(self):
        """Interactive mode for testing JavaScript code generation"""
        print("Starting interactive JavaScript code generation mode")
        print("Enter your natural language request or type 'exit' to quit")
        print("Note: Generated code will always return an array of records")

        while True:
            query = input("\nRequest: ")
            if query.lower() in ['exit', 'quit', 'q']:
                break

            try:
                result = self.generate_js_code(query)
                js_code = result["javascript_code"]

                print(f"\nGenerated JavaScript Code:\n")
                print(js_code)

                if not self.validate_js(js_code):
                    print("\nWarning: The generated code may have issues with returning an array of records.")
                    print("Make sure to add a proper return statement that returns the array.")
                else:
                    print("\n✓ Code successfully returns an array of records")

                print("\nContext used:")
                for model in result["context_used"]["models"]:
                    print(f"- Model: {model.get('name', '')} ({model.get('alias', '')})")

                print("\nKey API methods:")
                for api in result["context_used"]["api_methods"]:
                    print(f"- {api.get('name', '')}")

            except Exception as e:
                print(f"Error generating JavaScript code: {str(e)}")

    def api_generate_js_code(self, request_data: Dict) -> Dict:
        """API endpoint to generate JavaScript code from natural language query
        
        Args:
            request_data: Dictionary containing at minimum a 'query' field
            
        Returns:
            Dictionary with generated client and server scripts and metadata
        """
        try:
            # Extract the query from request data
            query = request_data.get('query', '')
            if not query:
                return {'error': 'No query provided', 'status': 'error'}
            
            # Optional parameters
            max_tokens = request_data.get('max_tokens', 1024)
            temperature = request_data.get('temperature', 0.2)
            
            # Ensure query emphasizes returning records if not already mentioned
            enhanced_query = query
            if "return" not in query.lower() and "array" not in query.lower():
                enhanced_query = f"{query} and return the array of records"

            # Retrieve relevant context using RagManager with emphasis on report examples
            context = self.rag_manager.hybrid_search(enhanced_query, k=25)
            print(f"Retrieved {len(context)} context items for query: '{enhanced_query}'")

            # Create JS generation prompt for client/server scripts
            prompt = self._create_js_prompt(enhanced_query, context)
            
            # Generate response from LLM
            response_text = self.llm_processor.generate_response(prompt, max_tokens=max_tokens, temperature=temperature)

            # Extract client and server scripts from response
            scripts = self._extract_client_server_scripts(response_text)
            
            # Ensure server script returns array of records if it doesn't already
            if scripts['server_script']:
                scripts['server_script'] = self._ensure_code_returns_array(scripts['server_script'])
            
            # Return formatted response
            return {
                'status': 'success',
                'query': query,
                'client_script': scripts['client_script'],
                'server_script': scripts['server_script'],
                'is_valid': self.validate_js(scripts['server_script']) if scripts['server_script'] else False,
                'context': {
                    'models_used': [
                        {'name': model.get('name', ''), 'alias': model.get('alias', '')} 
                        for model in context if model.get('type') == 'model'
                    ][:3],
                    'report_examples_used': [
                        {'name': example.get('name', '')} 
                        for example in context if example.get('type') == 'report_example'
                    ][:3],
                    'api_methods_used': [
                        {'name': api.get('name', '')} 
                        for api in context if api.get('type') == 'js_api'
                    ][:5]
                }
            }
        except Exception as e:
            return {
                'status': 'error',
                'error': str(e),
                'query': request_data.get('query', '')
            }
