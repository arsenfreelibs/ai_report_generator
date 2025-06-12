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

        # Format report examples
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
                    server_lines = server_script.split('\n')  # First 15 lines
                    report_examples_str += f"Server Script Pattern:\n```javascript\n"
                    report_examples_str += "\n".join(server_lines)
                    report_examples_str += "\n```\n"
                
                if client_script:
                    # Extract key patterns from client script
                    client_lines = client_script.split('\n')
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

        # Construct the system prompt with stronger emphasis on array_string fields
        system_prompt = """You are an expert JavaScript code generator for the SL2 system. Your task is to convert natural language requests into precise JavaScript code that uses SL2's API.

IMPORTANT:
- Your code MUST ALWAYS return an array of records as the final result.
- ALWAYS use model and field aliases in your code, NOT their display names.
- For array_string fields, you MUST use the KEY values in the filter (e.g., 'New', 'Open'), NOT the display values.

## REQUIRED SERVER SCRIPT STRUCTURE:
Your server script MUST follow this exact structure:

```javascript
async function(scope) {
  // Your data retrieval logic here
  const records = await p.iterMap(scope.find({}), record => record.attributes);

  return {
    main: records
  };
}
```

Follow these guidelines:
1. Only use the provided models, fields, and API methods in your code
2. Always refer to models and fields by their aliases (e.g., use 'test_db_1' not 'Test DB 1')
3. When calling p.getModel(), always use the model's alias (e.g., p.getModel('test_db_1'))
4. When querying fields, always use the field's alias (e.g., {status: 'active'}, not {Status: 'active'})
5. When querying array_string fields, use EXACTLY the key values provided (e.g., {status: 'New'}, not {status: 'New Status'})
6. For filtering records from the past week, use a date calculation
7. For limiting records, use .limit(N) method
8. Write async/await code where appropriate
9. Include error handling with try/catch blocks for robust code
10. Add helpful comments to explain your code
11. Make sure your code returns an object with 'main' property containing the records array
12. Return only the JavaScript code, with no additional explanations
13. Use the provided report examples as reference patterns when applicable

## COMMON MISTAKES TO AVOID:

INCORRECT EXAMPLE - DO NOT DO THIS:
```javascript
// WRONG: Using non-existent values for array_string fields
const notClosedWorkOrders = await workOrderModel.find({ status: 'not_closed' }).raw();
// This is WRONG because 'not_closed' is not a valid key in the status field
```

CORRECT APPROACH FOR "NOT CLOSED" STATUS:
```javascript
// RIGHT: Use $in operator with all valid status keys except 'closed'
const notClosedWorkOrders = await workOrderModel.find({ 
  status: { 
    $in: ['pending_assignment', 'assigned', 'in_progress', 'pause', 'completed', 'rejected', 'db_verification', 'for_closure', 'canceled', 'pause_requested'] 
  } 
}).raw();

// OR use $ne (not equal) to exclude 'closed' status
const notClosedWorkOrders = await workOrderModel.find({ status: { $ne: 'closed' } }).raw();
```

Remember: Always check the "Possible values" for array_string fields and use ONLY those exact keys!
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

## User Request:
{query}

## Task:
Generate JavaScript code that fulfills the user's request using the available models, fields, and API methods.

CRITICAL REQUIREMENTS:
1. Your server script MUST follow the required structure: async function(scope) {{ ... return {{ main: records }}; }}
2. ALWAYS use model aliases (e.g., 'test_db_1') instead of display names ('Test DB 1')
3. ALWAYS use field aliases (e.g., 'status') instead of display names ('Status')
4. For array_string fields, ALWAYS use the EXACT key values (e.g., 'New', 'Open'), NOT the display values
5. If filtering for "active" status, check the available values first and use the appropriate key
6. Use the report examples as reference patterns when applicable
7. DO NOT invent or guess field values - only use the exact keys provided in the field definitions
8. Return data in the format {{ main: records }} where records is an array

## JavaScript Code:
```javascript
"""

        print(f"Generated prompt: {prompt}")
        return prompt

    def _extract_js_from_response(self, response_text: str) -> str:
        """Extract JavaScript code from model response text"""
        # Look for code between JavaScript or JS code blocks
        js_pattern = r"```(?:javascript|js)?\s*([\s\S]*?)```"
        match = re.search(js_pattern, response_text)

        if match:
            return match.group(1).strip()

        # If no code found with pattern, look for anything that looks like JS code
        # This is a fallback in case the model doesn't format its response with code blocks
        js_like_pattern = r"(?:const|let|var|async function|function|\(\)\s*=>|await)\s+\w+.*?[;{]"
        matches = re.findall(js_like_pattern, response_text, re.MULTILINE)

        if matches:
            # Try to extract a coherent code block
            # Starting from the first match, include everything until end or clear non-code text
            start_idx = response_text.find(matches[0])
            code_section = response_text[start_idx:]

            # Look for clear end markers
            end_markers = ["## ", "Note:", "Explanation:", "This code"]
            for marker in end_markers:
                marker_idx = code_section.find(marker)
                if marker_idx > 0:
                    code_section = code_section[:marker_idx]

            return code_section.strip()

        # If still no code found, return the whole response
        return response_text.strip()

    def _ensure_code_returns_array(self, js_code: str) -> str:
        """Ensure the JavaScript code returns an array of records and uses aliases"""

        # Check if code already has a return statement
        if "return " in js_code:
            return js_code

        # Look for common patterns that should return records
        result_patterns = [
            r"const\s+(\w+)\s*=\s*await.*?model\.find\(.*?\);",
            r"const\s+(\w+)\s*=\s*await.*?p\.uiUtils\.fetchRecords\(.*?\);",
            r"let\s+(\w+)\s*=\s*await.*?model\.find\(.*?\);",
            r"let\s+(\w+)\s*=\s*await.*?p\.uiUtils\.fetchRecords\(.*?\);"
        ]

        for pattern in result_patterns:
            match = re.search(pattern, js_code, re.DOTALL)
            if match:
                var_name = match.group(1)

                # Handle fetchRecords which returns an object with data property
                if "fetchRecords" in match.group(0):
                    # Check if code already accesses .data property
                    if f"{var_name}.data" in js_code:
                        if not js_code.strip().endswith(';'):
                            js_code += ";\n"
                        js_code += f"\n// Return the array of records\nreturn {var_name}.data;";
                    else:
                        if not js_code.strip().endswith(';'):
                            js_code += ";\n"
                        js_code += f"\n// Return the array of records\nreturn {var_name}.data;";
                else:
                    if not js_code.strip().endswith(';'):
                        js_code += ";\n"
                    js_code += f"\n// Return the array of records\nreturn {var_name};"

                return js_code

        # If no pattern matched, add a general comment about returning records
        if not js_code.strip().endswith(';'):
            js_code += ";\n"
        js_code += "\n// IMPORTANT: Remember to return the array of records at the end of your function"

        return js_code

    def _create_client_js_prompt(self, query: str, context: List[Dict], server_code: str) -> str:
        """Create a prompt for client-side JS code generation with retrieved context"""
        # Categorize context items
        models_info = [item for item in context if item['type'] == 'model']
        report_examples = [item for item in context if item['type'] == 'report_example']
        
        # Format report examples focusing on client scripts
        client_examples_str = ""
        if report_examples:
            client_examples_str = "\n## Client-Side Report Examples:\n"
            for example in report_examples:
                client_examples_str += f"\n### Example: {example.get('name', 'Unknown')}\n"
                client_examples_str += f"User Request: {example.get('user_request', '')[:200]}...\n"
                
                # Focus on client script patterns
                client_script = example.get('client_script', '')
                if client_script:
                    client_lines = client_script.split('\n')[:30]  # First 30 lines for context
                    client_examples_str += f"Client Script Pattern:\n```javascript\n"
                    client_examples_str += "\n".join(client_lines)
                    client_examples_str += "\n```\n"

        # Format model information
        models_str = "\n".join([
            f"- Model: {item.get('name', '')} (alias: '{item.get('alias', '')}', id: {item.get('id', '')})"
            for item in models_info
        ])

        # Construct the system prompt for client-side code
        system_prompt = """You are an expert JavaScript client-side code generator for the SL2 system's amCharts visualizations. Your task is to convert natural language requests into client-side JavaScript code that creates interactive charts and visualizations.

## REQUIRED CLIENT SCRIPT STRUCTURE:
Your client script MUST follow this exact structure:

```javascript
function(chartdiv, scope) {
  const chart = am4core.create(chartdiv, am4charts.XYChart);

  chart.data = scope.main;

  return chart;
}
```

IMPORTANT GUIDELINES:
1. Generate ONLY client-side JavaScript code for amCharts visualization
2. The code should be a function that takes (chartdiv, scope) parameters
3. Use scope.main to access data provided by the server script
4. Always use amCharts libraries (am4core, am5, am5xy, am5percent, etc.)
5. Create interactive tooltips with clickable links where appropriate
6. Include proper event handlers for user interactions
7. Use responsive design principles
8. Return the chart/root object at the end of the function
9. Write clean, commented code explaining key visualization logic
10. Handle empty data gracefully
11. Use appropriate chart types based on the request (pie, bar, line, etc.)
12. Include animations and transitions for better UX

Client-side code structure:
- Should be a function(chartdiv, scope) { ... return chart/root; }
- Access server data via scope.main
- Create amCharts visualizations
- Add interactivity and tooltips
- Handle responsive behavior
"""

        # Construct the full prompt
        prompt = f"""
{system_prompt}

## Available Models (for context):
{models_str}

{client_examples_str}

## Server Code Context:
The server script returns data in this structure:
```javascript
{server_code}
```

## User Request:
{query}

## Task:
Generate client-side JavaScript code that creates an interactive amCharts visualization based on the user's request and the data structure provided by the server code.

CRITICAL REQUIREMENTS:
1. Your client script MUST follow the required structure: function(chartdiv, scope) { ... return chart; }
2. Use scope.main to access the data from server
3. Create appropriate amCharts visualization
4. Add interactive tooltips with clickable links when relevant
5. Use responsive design and proper styling
6. Include animations and smooth transitions
7. Handle edge cases (empty data, etc.)
8. Follow amCharts best practices

## Client-Side JavaScript Code:
```javascript
"""

        print(f"Client prompt: {prompt}")
        return prompt

    def generate_server_script_prompt(self, query: str) -> str:
        """Process a natural language query and return server script prompt"""
        # Ensure query emphasizes returning records if not already mentioned
        enhanced_query = query
        if "return" not in query.lower() and "array" not in query.lower():
            enhanced_query = f"{query} and return the array of records"

        # Retrieve relevant context
        context = self.rag_manager.hybrid_search(enhanced_query, k=15)
        print(f"Retrieved {len(context)} context items for query: '{enhanced_query}'")

        # Create JS generation prompt
        prompt = self._create_js_prompt(enhanced_query, context)
        # print(f"prompt: {prompt}")   
        return prompt

    def generate_js_code(self, query: str) -> Dict:
        """Process a natural language query and return both server and client JavaScript code"""
        # Ensure query emphasizes returning records if not already mentioned
        enhanced_query = query
        if "return" not in query.lower() and "array" not in query.lower():
            enhanced_query = f"{query} and return the array of records"

        # Retrieve relevant context using RagManager
        context = self.rag_manager.hybrid_search(enhanced_query, k=25)
        print(f"Retrieved {len(context)} context items for query: '{enhanced_query}'")

        # Create server-side JS generation prompt
        server_prompt = self._create_js_prompt(enhanced_query, context)
        # print(f"Server prompt: {server_prompt}")
        
        # Generate server-side response from LLM
        server_response_text = self.llm_processor.generate_response(server_prompt, max_tokens=1024, temperature=0.2)
        print(f"Server response: {server_response_text}")

        # Extract server JS from response
        server_js_code = self._extract_js_from_response(server_response_text)

        # Ensure server code returns array of records if it doesn't already
        server_js_code = self._ensure_code_returns_array(server_js_code)

        # Create client-side JS generation prompt
        client_prompt = self._create_client_js_prompt(enhanced_query, context, server_js_code)
        # print(f"Client prompt: {client_prompt}")
        
        # Generate client-side response from LLM
        client_response_text = self.llm_processor.generate_response(client_prompt, max_tokens=1024, temperature=0.2)
        print(f"Client response: {client_response_text}")

        # Extract client JS from response
        client_js_code = self._extract_js_from_response(client_response_text)

        # Return combined result
        return {
            "natural_language_query": query,
            "server_script": server_js_code,
            "client_script": client_js_code,
            "context_used": {
                "models": [item for item in context if item['type'] == 'model'][:3],
                "fields": [item for item in context if item['type'] == 'field'][:5],
                "api_methods": [item for item in context if item['type'] == 'js_api'][:5],
                "report_examples": [item for item in context if item['type'] == 'report_example'][:3],
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

    def validate_client_js(self, js_code: str) -> bool:
        """Basic validation of client-side JavaScript code"""
        # Check for amCharts patterns
        has_amcharts = "am4core" in js_code or "am5" in js_code
        has_function = "function" in js_code and "chartdiv" in js_code and "scope" in js_code
        has_return = "return " in js_code
        has_chart_creation = "create" in js_code or "Container" in js_code or "Chart" in js_code
        
        return has_amcharts and has_function and has_return and has_chart_creation

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
            Dictionary with generated code and metadata
        """
        try:
            # Extract the query from request data
            query = request_data.get('query', '')
            if not query:
                return {'error': 'No query provided', 'status': 'error'}
            
            # Optional parameters (for future use)
            max_tokens = request_data.get('max_tokens', 1024)
            temperature = request_data.get('temperature', 0.2)
            
            # Generate code using the main method
            result = self.generate_js_code(query)
            
            # Validate both server and client code
            server_valid = self.validate_js(result['server_script'])
            client_valid = self.validate_client_js(result['client_script'])
            
            # Return formatted response
            return {
                'status': 'success',
                'query': result['natural_language_query'],
                'server_script': result['server_script'],
                'client_script': result['client_script'],
                'validation': {
                    'server_valid': server_valid,
                    'client_valid': client_valid,
                    'overall_valid': server_valid and client_valid
                },
                'context': {
                    'models_used': [
                        {'name': model.get('name', ''), 'alias': model.get('alias', '')} 
                        for model in result['context_used']['models']
                    ],
                    'api_methods_used': [
                        {'name': api.get('name', '')} 
                        for api in result['context_used']['api_methods']
                    ],
                    'report_examples_used': [
                        {'name': example.get('name', '')} 
                        for example in result['context_used']['report_examples']
                    ]
                }
            }
        except Exception as e:
            return {
                'status': 'error',
                'error': str(e),
                'query': request_data.get('query', '')
            }
