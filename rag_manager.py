import logging
from typing import Dict, List
from sentence_transformers import CrossEncoder
from config import RERANKER_MODEL, GDRIVE_FILE_ID, GDRIVE_CREDENTIALS_PATH
from metadata_indexer import MetadataIndexer
from js_api_indexer import JsApiIndexer
from report_example_indexer import ReportExampleIndexer
from report_data_example_indexer import ReportDataExampleIndexer
from report_chart_example_indexer import ReportChartExampleIndexer

logger = logging.getLogger(__name__)

class RagManager:
    """Manages hybrid search across all indexers (metadata, JS API, report examples, report data examples, and chart examples)"""

    def __init__(self, metadata_path: str, reranker_model_name: str = RERANKER_MODEL, 
                 gdrive_file_id: str = GDRIVE_FILE_ID, credentials_path: str = GDRIVE_CREDENTIALS_PATH):
        """Initialize the RAG manager"""
        self.metadata_indexer = MetadataIndexer(metadata_path)
        self.js_api_indexer = JsApiIndexer(
            gdrive_file_id=gdrive_file_id,
            credentials_path=credentials_path
        )
        self.report_example_indexer = ReportExampleIndexer()
        self.report_data_example_indexer = ReportDataExampleIndexer()
        self.report_chart_example_indexer = ReportChartExampleIndexer()
        self.reranker_model_name = reranker_model_name
        self.reranker = None

    def initialize(self):
        """Initialize all indexers"""
        self.metadata_indexer.load_metadata()
        self.metadata_indexer.create_indexes()
        self.js_api_indexer.create_indexes()
        self.report_example_indexer.create_indexes()
        self.report_data_example_indexer.create_indexes()
        self.report_chart_example_indexer.create_indexes()
        
        # Initialize reranker
        self.reranker = CrossEncoder(self.reranker_model_name)
        
        logger.info("RAG Manager initialized with all indexers")

    def hybrid_search(self, query: str, k: int = 20) -> List[Dict]:
        """Search across all indexers using hybrid retrieval"""
        print(f"Starting hybrid search for query: '{query}', k={k}")
        
        # Ensure indexers are initialized
        if not hasattr(self.metadata_indexer, 'vector_store') or self.metadata_indexer.vector_store is None:
            self.initialize()

        # Split k between models, JS API, report examples, report data examples, and chart examples
        model_k = int(k * 0.5)
        js_api_k = int(k * 0.2)
        # report_k = int(k * 0.15)
        report_data_k = int(k * 0.15)
        chart_k = int(k * 0.15)

        # Search models
        model_results = self.metadata_indexer.search(query, k=model_k)
        print(f"Model search returned {len(model_results)} documents")

        # Search JS API
        js_api_results = self.js_api_indexer.search(query, k=js_api_k)
        print(f"JS API search returned {len(js_api_results)} documents")

        # Search report examples
        report_results = self.report_example_indexer.search(query, k=report_k)
        print(f"Report example search returned {len(report_results)} documents")

        # Search report data examples (for server scripts)
        report_data_results = self.report_data_example_indexer.search(query, k=report_data_k)
        print(f"Report data example search returned {len(report_data_results)} documents")

        # Search chart examples (for client scripts)
        chart_results = self.report_chart_example_indexer.search(query, k=chart_k)
        print(f"Chart example search returned {len(chart_results)} documents")

        # Combine results
        combined_docs = model_results + js_api_results + report_results + report_data_results + chart_results

        print(f"Combined search returned {len(combined_docs)} unique documents")
        return combined_docs

    def get_models_context(self, query: str, k: int = 10) -> List[Dict]:
        """Get only model-related context"""
        return self.metadata_indexer.search(query, k=k)

    def get_js_api_context(self, query: str, k: int = 10) -> List[Dict]:
        """Get only JS API-related context"""
        return self.js_api_indexer.search(query, k=k)

    def get_report_examples_context(self, query: str, k: int = 5) -> List[Dict]:
        """Get only report example-related context"""
        return self.report_example_indexer.search(query, k=k)

    def get_report_data_examples_context(self, query: str, k: int = 5) -> List[Dict]:
        """Get only report data example-related context"""
        return self.report_data_example_indexer.search(query, k=k)

    def get_chart_examples_context(self, query: str, k: int = 5) -> List[Dict]:
        """Get only chart example-related context"""
        return self.report_chart_example_indexer.search(query, k=k)

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
        self.report_example_indexer.create_indexes()
        self.report_data_example_indexer.create_indexes()
        self.report_chart_example_indexer.create_indexes()
        
        logger.info("All indexes refreshed")
