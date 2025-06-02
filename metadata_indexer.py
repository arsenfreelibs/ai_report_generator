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
from config import EMBEDDING_MODEL, RERANKER_MODEL
from metadata_loader import MetadataLoader

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

class MetadataIndexer:
    """Handles indexing and retrieval of SL2 model metadata only"""

    def __init__(self, metadata_path: str, embedding_model_name: str = EMBEDDING_MODEL):
        """Initialize the metadata indexer"""
        self.metadata_path = metadata_path
        self.embedding_model_name = embedding_model_name
        self.metadata_loader = MetadataLoader(metadata_path)
        self.vector_store = None
        self.index_data = []
        self.bm25_index = None
        self.raw_texts = []

    @property
    def metadata(self):
        """Access metadata through the loader"""
        return self.metadata_loader.metadata

    @property
    def fields_by_model(self):
        """Access fields_by_model through the loader"""
        return self.metadata_loader.fields_by_model

    def load_metadata(self) -> Dict:
        """Load metadata using the MetadataLoader"""
        return self.metadata_loader.load_metadata()

    def extract_index_data(self) -> List[Dict]:
        """Extract models from metadata for indexing"""
        index_data = []

        # First ensure metadata is loaded
        if not self.metadata:
            self.load_metadata()

        # Process models only
        for model in self.metadata.get('data', {}).get('model', []):
            model_entry = {
                'type': 'model',
                'id': model.get('id'),
                'alias': model.get('alias', ''),
                'name': model.get('name', ''),
                'content': f"Model: {model.get('name', '')} (alias: {model.get('alias', '')}, id: {model.get('id')})",
            }
            index_data.append(model_entry)

        self.index_data = index_data
        self.raw_texts = [item['content'] for item in index_data]

        return index_data

    def create_indexes(self):
        """Create vector store and BM25 index for models"""
        if not self.index_data:
            self.extract_index_data()

        # Create embeddings using sentence-transformers
        embedding_model = SentenceTransformerEmbeddings(model_name=self.embedding_model_name)

        # Create FAISS index for models
        docs = [item['content'] for item in self.index_data]
        metadatas = [{k: v for k, v in item.items() if k != 'content'} for item in self.index_data]
        self.vector_store = FAISS.from_texts(docs, embedding_model, metadatas=metadatas)

        # Create BM25 index
        tokenized_corpus = [doc.lower().split() for doc in self.raw_texts]
        self.bm25_index = BM25Okapi(tokenized_corpus)

        logger.info(f"Created model indexes with {self.vector_store.index.ntotal} models")

        return self.vector_store

    def search(self, query: str, k: int = 10) -> List[Dict]:
        """Search models using hybrid retrieval"""
        if not hasattr(self, 'vector_store') or self.vector_store is None:
            self.create_indexes()

        # Dense retrieval with FAISS
        dense_results = self.vector_store.similarity_search(query, k=k)
        dense_documents = [doc.metadata for doc in dense_results]

        # Sparse retrieval with BM25
        tokenized_query = query.lower().split()
        bm25_scores = self.bm25_index.get_scores(tokenized_query)
        top_n = min(k, len(self.raw_texts))
        top_indices = np.argsort(bm25_scores)[-top_n:][::-1]
        sparse_documents = [self.index_data[i] for i in top_indices]

        # Combine results (removing duplicates)
        combined_docs = []
        seen_ids = set()

        for doc in dense_documents + sparse_documents:
            doc_id = f"{doc.get('type', '')}-{doc.get('id', '')}-{doc.get('alias', '')}"
            if doc_id not in seen_ids:
                seen_ids.add(doc_id)
                
                # Enrich model with fields
                model_id = doc.get('id')
                model_alias = doc.get('alias', '')
                
                fields = []
                if model_id in self.fields_by_model:
                    fields = self.fields_by_model[model_id]
                elif model_alias in self.fields_by_model:
                    fields = self.fields_by_model[model_alias]
                
                enriched_doc = doc.copy()
                enriched_doc['fields'] = fields
                combined_docs.append(enriched_doc)

        return combined_docs[:k]
