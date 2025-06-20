import os
import numpy as np
import logging
from typing import Dict, List
from langchain_community.vectorstores import FAISS
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from rank_bm25 import BM25Okapi
from config import EMBEDDING_MODEL

logger = logging.getLogger(__name__)

class ReportDataExampleIndexer:
    """Handles indexing and retrieval of SL2 report data examples"""

    def __init__(self, knowledge_path: str = "knowledge/report_data_examples", embedding_model_name: str = EMBEDDING_MODEL):
        """Initialize the report data example indexer"""
        self.knowledge_path = knowledge_path
        self.embedding_model_name = embedding_model_name
        self.vector_store = None
        self.index_data = []
        self.bm25_index = None
        self.raw_texts = []

    def _parse_report_data_file(self, file_path: str) -> Dict:
        """Parse a report data example file and extract sections"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            # Extract sections using markers
            sections = {}
            
            # Extract user_request
            user_request_start = content.find('<user_request>')
            user_request_end = content.find('</user_request>')
            if user_request_start != -1 and user_request_end != -1:
                sections['user_request'] = content[user_request_start + 14:user_request_end].strip()

            # Extract description
            desc_start = content.find('<description>')
            desc_end = content.find('</description>')
            if desc_start != -1 and desc_end != -1:
                sections['description'] = content[desc_start + 13:desc_end].strip()

            # Extract server_script (focus on server scripts for data examples)
            server_start = content.find('<server_script>')
            server_end = content.find('</server_script>')
            if server_start != -1 and server_end != -1:
                sections['server_script'] = content[server_start + 15:server_end].strip()

            # Extract client_script if present
            client_start = content.find('<client_script>')
            client_end = content.find('</client_script>')
            if client_start != -1 and client_end != -1:
                sections['client_script'] = content[client_start + 15:client_end].strip()

            return sections

        except Exception as e:
            logger.error(f"Error parsing report data file {file_path}: {str(e)}")
            return {}

    def extract_index_data(self) -> List[Dict]:
        """Extract report data examples from knowledge directory for indexing"""
        index_data = []

        if not os.path.exists(self.knowledge_path):
            logger.warning(f"Knowledge path {self.knowledge_path} does not exist")
            return index_data

        # Process all files in the report_data_examples directory
        for filename in os.listdir(self.knowledge_path):
            file_path = os.path.join(self.knowledge_path, filename)
            
            # Skip directories
            if os.path.isdir(file_path):
                continue

            sections = self._parse_report_data_file(file_path)
            
            if not sections:
                continue

            # Create indexable content combining all sections with emphasis on server scripts
            content_parts = []
            
            if sections.get('user_request'):
                content_parts.append(f"User Request: {sections['user_request']}")
            
            if sections.get('description'):
                content_parts.append(f"Description: {sections['description']}")

            # Add server script content for better matching
            if sections.get('server_script'):
                # Extract key patterns from server script for indexing
                server_script = sections['server_script']
                # Add keywords from server script
                if 'find(' in server_script:
                    content_parts.append("Server Script: Data retrieval with find queries")
                if 'filter(' in server_script:
                    content_parts.append("Server Script: Data filtering operations")
                if 'map(' in server_script:
                    content_parts.append("Server Script: Data transformation with map")
                if 'aggregate(' in server_script:
                    content_parts.append("Server Script: Data aggregation operations")

            content = " ".join(content_parts)

            if content.strip():
                report_entry = {
                    'type': 'report_data_example',
                    'name': filename,
                    'user_request': sections.get('user_request', ''),
                    'description': sections.get('description', ''),
                    'server_script': sections.get('server_script', ''),
                    'client_script': sections.get('client_script', ''),
                    'content': content
                }
                index_data.append(report_entry)

        self.index_data = index_data
        self.raw_texts = [item['content'] for item in index_data]

        logger.info(f"Extracted {len(index_data)} report data examples for indexing")
        return index_data

    def create_indexes(self):
        """Create vector store and BM25 index for report data examples"""
        if not self.index_data:
            self.extract_index_data()

        if not self.index_data:
            logger.warning("No report data examples found to index")
            return None

        # Create embeddings using sentence-transformers
        embedding_model = SentenceTransformerEmbeddings(model_name=self.embedding_model_name)

        # Create FAISS index for report data examples
        docs = [item['content'] for item in self.index_data]
        metadatas = [{k: v for k, v in item.items() if k != 'content'} for item in self.index_data]
        self.vector_store = FAISS.from_texts(docs, embedding_model, metadatas=metadatas)

        # Create BM25 index
        tokenized_corpus = [doc.lower().split() for doc in self.raw_texts]
        self.bm25_index = BM25Okapi(tokenized_corpus)

        logger.info(f"Created report data example indexes with {self.vector_store.index.ntotal} examples")
        return self.vector_store

    def search(self, query: str, k: int = 5) -> List[Dict]:
        """Search report data examples using hybrid retrieval"""
        if not hasattr(self, 'vector_store') or self.vector_store is None:
            self.create_indexes()

        if not self.vector_store:
            return []

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
        seen_names = set()

        for doc in dense_documents + sparse_documents:
            doc_name = doc.get('name', '')
            if doc_name not in seen_names:
                seen_names.add(doc_name)
                combined_docs.append(doc)

        return combined_docs[:k]
