import logging
from typing import Dict, List
from sentence_transformers import CrossEncoder
from config import RERANKER_MODEL
from metadata_indexer import MetadataIndexer
from js_api_indexer import JsApiIndexer

logger = logging.getLogger(__name__)

class RagManager:
    """Manages hybrid search across all indexers (metadata and JS API)"""

    def __init__(self, metadata_path: str, reranker_model_name: str = RERANKER_MODEL):
        """Initialize the RAG manager"""
        self.metadata_indexer = MetadataIndexer(metadata_path)
        self.js_api_indexer = JsApiIndexer()
        self.reranker_model_name = reranker_model_name
        self.reranker = None

    def initialize(self):
        """Initialize all indexers"""
        self.metadata_indexer.load_metadata()
        self.metadata_indexer.create_indexes()
        self.js_api_indexer.create_indexes()
        
        # Initialize reranker
        self.reranker = CrossEncoder(self.reranker_model_name)
        
        logger.info("RAG Manager initialized with all indexers")

    def hybrid_search(self, query: str, k: int = 20) -> List[Dict]:
        """Search across all indexers using hybrid retrieval"""
        print(f"Starting hybrid search for query: '{query}', k={k}")
        
        # Ensure indexers are initialized
        if not hasattr(self.metadata_indexer, 'vector_store') or self.metadata_indexer.vector_store is None:
            self.initialize()

        # Split k between models and JS API (give more weight to models)
        model_k = int(k * 0.7)  # 70% for models
        js_api_k = int(k * 0.5)  # 50% for JS API

        # Search models
        model_results = self.metadata_indexer.search(query, k=model_k)
        print(f"Model search returned {len(model_results)} documents")

        # Search JS API
        js_api_results = self.js_api_indexer.search(query, k=js_api_k)
        print(f"JS API search returned {len(js_api_results)} documents")

        # Combine results
        combined_docs = model_results + js_api_results

        print(f"Combined search returned {len(combined_docs)} unique documents")

        return combined_docs

    def get_models_context(self, query: str, k: int = 10) -> List[Dict]:
        """Get only model-related context"""
        return self.metadata_indexer.search(query, k=k)

    def get_js_api_context(self, query: str, k: int = 10) -> List[Dict]:
        """Get only JS API-related context"""
        return self.js_api_indexer.search(query, k=k)

    def update_metadata(self, new_metadata: Dict, action: str = 'replace') -> Dict:
        """Update metadata and rebuild indexes
        
        Args:
            new_metadata: New metadata to add or replace
            action: 'add' to append new models, 'replace' to replace existing
            
        Returns:
            Dictionary with update results
        """
        # Update metadata through the indexer
        result = self.metadata_indexer.metadata_loader.update_metadata(new_metadata, action)
        
        # Rebuild indexes with updated metadata
        self.metadata_indexer.extract_index_data()
        self.metadata_indexer.create_indexes()
        
        logger.info(f"Metadata updated and indexes rebuilt: {result['message']}")
        
        return result
    
    def refresh_indexes(self):
        """Refresh all indexes (useful after metadata changes)"""
        self.metadata_indexer.extract_index_data()
        self.metadata_indexer.create_indexes()
        self.js_api_indexer.create_indexes()
        
        logger.info("All indexes refreshed")
