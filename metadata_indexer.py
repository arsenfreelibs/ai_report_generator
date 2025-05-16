import os
import json
import numpy as np
import logging
import sys
from typing import Dict, List, Optional, Union, Any
from sentence_transformers import SentenceTransformer, CrossEncoder
from langchain_community.vectorstores import FAISS
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from rank_bm25 import BM25Okapi

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

class MetadataIndexer:
    """Handles indexing and retrieval of SL2 metadata with focus on models and JS API"""

    def __init__(self, metadata_path: str, embedding_model_name: str = "all-MiniLM-L6-v2",
                 reranker_model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        """Initialize the metadata indexer"""
        self.metadata_path = metadata_path
        self.embedding_model_name = embedding_model_name
        self.reranker_model_name = reranker_model_name
        self.metadata = None
        self.vector_store = None
        self.model_index_data = []  # For models only
        self.js_api_index_data = []  # For JS API only
        self.fields_by_model = {}  # Quick lookup for fields by model
        self.bm25_index = None
        self.js_api_bm25_index = None
        self.model_bm25_index = None
        self.reranker = None
        self.raw_model_texts = []
        self.raw_js_api_texts = []

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

    def extract_index_data(self) -> tuple:
        """Extract models and JS API methods from metadata for indexing"""
        model_index_data = []
        js_api_index_data = []

        # First ensure metadata is loaded
        if not self.metadata:
            self.load_metadata()

        # Process models only
        for model in self.metadata.get('data', {}).get('model', []):
            # Add model info
            model_entry = {
                'type': 'model',
                'id': model.get('id'),
                'alias': model.get('alias', ''),
                'name': model.get('name', ''),
                'content': f"Model: {model.get('name', '')} (alias: {model.get('alias', '')}, id: {model.get('id')})",
            }
            model_index_data.append(model_entry)

        # Add JS API methods from SL2 system documentation
        js_api_methods = [
            {
                'type': 'js_api',
                'name': 'p.getModel',
                'description': 'Gets a model by alias or ID',
                'syntax': 'p.getModel(model_alias_or_id)',
                'example': "const model = await p.getModel('test_db_1');",
                'content': "JS API: p.getModel(model_alias_or_id) - Gets a model by alias or ID. Example: const model = await p.getModel('test_db_1');"
            },
            {
                'type': 'js_api',
                'name': 'model.find',
                'description': 'Queries records from the model with a filter',
                'syntax': 'model.find(filter_object)',
                'example': "const records = await model.find({status: 'active'});",
                'content': "JS API: model.find(filter_object) - Queries records from the model with a filter. Example: const records = await model.find({status: 'active'});"
            },
            {
                'type': 'js_api',
                'name': 'p.uiUtils.fetchRecords',
                'description': 'Fetches records from a model with advanced options',
                'syntax': 'p.uiUtils.fetchRecords(model_alias, params)',
                'example': "const result = await p.uiUtils.fetchRecords('test_db_1', {filter: 'status = \"active\"', fields: {id: 'id', name: 'name'}, page: {size: 10}});",
                'content': "JS API: p.uiUtils.fetchRecords(model_alias, params) - Fetches records from a model with advanced options. Example: const result = await p.uiUtils.fetchRecords('test_db_1', {filter: 'status = \"active\"', fields: {id: 'id', name: 'name'}, page: {size: 10}});"
            },
            {
                'type': 'js_api',
                'name': 'p.record.getField',
                'description': 'Gets field object from the current record',
                'syntax': 'p.record.getField(field_alias)',
                'example': "const field = p.record.getField('name');",
                'content': "JS API: p.record.getField(field_alias) - Gets field object from the current record. Example: const field = p.record.getField('name');"
            },
            {
                'type': 'js_api',
                'name': 'p.record.getValue',
                'description': 'Gets value of a field from the current record',
                'syntax': 'p.record.getValue(field_alias)',
                'example': "const value = p.record.getValue('name');",
                'content': "JS API: p.record.getValue(field_alias) - Gets value of a field from the current record. Example: const value = p.record.getValue('name');"
            },
            {
                'type': 'js_api',
                'name': 'p.record.setValue',
                'description': 'Sets value of a field in the current record',
                'syntax': 'p.record.setValue(field_alias, value)',
                'example': "p.record.setValue('name', 'New Name');",
                'content': "JS API: p.record.setValue(field_alias, value) - Sets value of a field in the current record. Example: p.record.setValue('name', 'New Name');"
            },
            {
                'type': 'js_api',
                'name': 'p.iterEach',
                'description': 'Iteratively processes records from a query',
                'syntax': 'p.iterEach(query, [batchSize,] callback)',
                'example': "await p.iterEach(model.find({status: 'active'}), (record) => { /* process each record */ });",
                'content': "JS API: p.iterEach(query, [batchSize,] callback) - Iteratively processes records from a query. Example: await p.iterEach(model.find({status: 'active'}), (record) => { /* process each record */ });"
            },
            {
                'type': 'js_api',
                'name': 'filtering operators',
                'description': 'Various operators for filtering records',
                'syntax': "field: {'operator': value}",
                'example': "model.find({count: {'>': 10}, name: {'LIKE': 'test%'}});",
                'content': "JS API: Filtering operators - Various operators for filtering records: '>' (greater than), '<' (less than), '>=' (greater or equal), '<=' (less or equal), '!=' (not equal), 'LIKE' (contains), 'STARTSWITH', 'ENDSWITH', 'IN' (in list), 'NOTIN' (not in list), 'ISNULL', 'ISNOTNULL'"
            },
        ]

        # Add API methods to index data
        js_api_index_data.extend(js_api_methods)

        self.model_index_data = model_index_data
        self.js_api_index_data = js_api_index_data

        # Extract raw texts for BM25 indexing
        self.raw_model_texts = [item['content'] for item in model_index_data]
        self.raw_js_api_texts = [item['content'] for item in js_api_index_data]

        return model_index_data, js_api_index_data

    def create_indexes(self):
        """Create separate vector stores and BM25 indexes for models and JS API"""
        if not self.model_index_data or not self.js_api_index_data:
            self.extract_index_data()

        # Create embeddings using sentence-transformers
        embedding_model = SentenceTransformerEmbeddings(model_name=self.embedding_model_name)

        # Create FAISS index for models
        model_docs = [item['content'] for item in self.model_index_data]
        model_metadatas = [{k: v for k, v in item.items() if k != 'content'} for item in self.model_index_data]
        self.model_vector_store = FAISS.from_texts(model_docs, embedding_model, metadatas=model_metadatas)

        # Create FAISS index for JS API
        js_api_docs = [item['content'] for item in self.js_api_index_data]
        js_api_metadatas = [{k: v for k, v in item.items() if k != 'content'} for item in self.js_api_index_data]
        self.js_api_vector_store = FAISS.from_texts(js_api_docs, embedding_model, metadatas=js_api_metadatas)

        logger.info(f"Created vector stores with {self.model_vector_store.index.ntotal} models and {self.js_api_vector_store.index.ntotal} JS API methods")

        # Create BM25 indexes
        # Tokenize the texts
        tokenized_model_corpus = [doc.lower().split() for doc in self.raw_model_texts]
        self.model_bm25_index = BM25Okapi(tokenized_model_corpus)

        tokenized_js_api_corpus = [doc.lower().split() for doc in self.raw_js_api_texts]
        self.js_api_bm25_index = BM25Okapi(tokenized_js_api_corpus)

        # Initialize reranker
        self.reranker = CrossEncoder(self.reranker_model_name)

        return self.model_vector_store, self.js_api_vector_store

    def hybrid_search(self, query: str, k: int = 20) -> List[Dict]:
        """Search metadata using hybrid retrieval for both models and JS API"""
        print(f"Starting hybrid search for query: '{query}', k={k}")
        if not hasattr(self, 'model_vector_store') or not hasattr(self, 'js_api_vector_store'):
            self.create_indexes()

        # Split k between models and JS API (give more weight to models)
        model_k = int(k * 0.7)  # 70% for models
        js_api_k = int(k * 0.5)  # 50% for JS API

        # Dense retrieval with FAISS for models
        model_dense_results = self.model_vector_store.similarity_search(query, k=model_k)
        model_dense_documents = [doc.metadata for doc in model_dense_results]
        print(f"Dense search returned {len(model_dense_documents)} model documents")

        # Dense retrieval with FAISS for JS API
        js_api_dense_results = self.js_api_vector_store.similarity_search(query, k=js_api_k)
        js_api_dense_documents = [doc.metadata for doc in js_api_dense_results]
        print(f"Dense search returned {len(js_api_dense_documents)} JS API documents")

        # Sparse retrieval with BM25 for models
        tokenized_query = query.lower().split()
        model_bm25_scores = self.model_bm25_index.get_scores(tokenized_query)
        top_n_models = min(model_k, len(self.raw_model_texts))
        top_model_indices = np.argsort(model_bm25_scores)[-top_n_models:][::-1]
        model_sparse_documents = [self.model_index_data[i] for i in top_model_indices]
        print(f"Sparse search returned {len(model_sparse_documents)} model documents")

        # Sparse retrieval with BM25 for JS API
        js_api_bm25_scores = self.js_api_bm25_index.get_scores(tokenized_query)
        top_n_js_api = min(js_api_k, len(self.raw_js_api_texts))
        top_js_api_indices = np.argsort(js_api_bm25_scores)[-top_n_js_api:][::-1]
        js_api_sparse_documents = [self.js_api_index_data[i] for i in top_js_api_indices]
        print(f"Sparse search returned {len(js_api_sparse_documents)} JS API documents")

        # Combine results (removing duplicates)
        combined_models = []
        seen_model_ids = set()

        for doc in model_dense_documents + model_sparse_documents:
            # Create a unique identifier for the model
            doc_id = f"{doc.get('type', '')}-{doc.get('id', '')}-{doc.get('alias', '')}"
            if doc_id not in seen_model_ids:
                seen_model_ids.add(doc_id)
                combined_models.append(doc)

        combined_js_api = []
        seen_js_api_ids = set()

        for doc in js_api_dense_documents + js_api_sparse_documents:
            # Create a unique identifier for the JS API
            doc_id = f"{doc.get('type', '')}-{doc.get('name', '')}"
            if doc_id not in seen_js_api_ids:
                seen_js_api_ids.add(doc_id)
                combined_js_api.append(doc)

        # Enrich models with their fields
        enriched_models = []
        for model in combined_models[:model_k]:  # Limit to top model_k models
            model_id = model.get('id')
            model_alias = model.get('alias', '')

            # Add fields for this model
            fields = []
            if model_id in self.fields_by_model:
                fields = self.fields_by_model[model_id]
            elif model_alias in self.fields_by_model:
                fields = self.fields_by_model[model_alias]

            enriched_model = model.copy()
            enriched_model['fields'] = fields
            enriched_models.append(enriched_model)

        # Combine enriched models and JS API methods
        combined_docs = enriched_models + combined_js_api[:js_api_k]

        print(f"Combined search returned {len(combined_docs)} unique documents")

        return combined_docs
