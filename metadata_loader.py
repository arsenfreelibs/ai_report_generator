import json
import logging
from typing import Dict, List

logger = logging.getLogger(__name__)

class MetadataLoader:
    """Handles loading and organizing SL2 metadata"""

    def __init__(self, metadata_path: str):
        """Initialize the metadata loader"""
        self.metadata_path = metadata_path
        self.metadata = None
        self.fields_by_model = {}  # Quick lookup for fields by model

    def load_metadata(self) -> Dict:
        """Load SL2 system metadata from JSON file"""
        try:
            with open(self.metadata_path, 'r') as f:
                self.metadata = json.load(f)
            logger.info(f"Successfully loaded metadata with {len(self.metadata['data']['model'])} models")

            # Organize fields by model for quick lookup
            for model in self.metadata.get('data', {}).get('model', []):
                model_id = model.get('id')
                model_alias = model.get('alias', '')
                
                # Process fields to extract options
                processed_fields = []
                for field in model.get('field', []):
                    processed_field = field.copy()
                    
                    # Parse options if it's a string
                    if isinstance(field.get('options'), str):
                        try:
                            options_dict = json.loads(field.get('options', '{}'))
                            processed_field['parsed_options'] = options_dict
                            
                            # Extract values for array_string fields
                            if field.get('type') == 'array_string' and 'values' in options_dict:
                                processed_field['values'] = options_dict.get('values', {})
                        except json.JSONDecodeError:
                            processed_field['parsed_options'] = {}
                    
                    processed_fields.append(processed_field)
                
                self.fields_by_model[model_id] = processed_fields
                self.fields_by_model[model_alias] = processed_fields

            return self.metadata
        except Exception as e:
            logger.error(f"Error loading metadata: {str(e)}")
            # Provide empty structure if file not found
            self.metadata = {"data": {"model": []}}
            self.fields_by_model = {}
            return self.metadata

    def get_fields_by_model(self, model_id: str) -> List[Dict]:
        """Get fields for a specific model by ID or alias"""
        return self.fields_by_model.get(model_id, [])

    def get_all_models(self) -> List[Dict]:
        """Get all models from metadata"""
        if not self.metadata:
            self.load_metadata()
        return self.metadata.get('data', {}).get('model', [])
