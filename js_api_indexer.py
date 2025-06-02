import numpy as np
import logging
from typing import Dict, List
from langchain_community.vectorstores import FAISS
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from rank_bm25 import BM25Okapi
from config import EMBEDDING_MODEL

logger = logging.getLogger(__name__)

class JsApiIndexer:
    """Handles indexing and retrieval of SL2 JavaScript API methods"""

    def __init__(self, embedding_model_name: str = EMBEDDING_MODEL):
        """Initialize the JS API indexer"""
        self.embedding_model_name = embedding_model_name
        self.vector_store = None
        self.index_data = []
        self.bm25_index = None
        self.raw_texts = []

    def extract_index_data(self) -> List[Dict]:
        """Extract JS API methods for indexing"""
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

        self.index_data = js_api_methods
        self.raw_texts = [item['content'] for item in js_api_methods]

        return js_api_methods

    def create_indexes(self):
        """Create vector store and BM25 index for JS API methods"""
        if not self.index_data:
            self.extract_index_data()

        # Create embeddings using sentence-transformers
        embedding_model = SentenceTransformerEmbeddings(model_name=self.embedding_model_name)

        # Create FAISS index for JS API
        docs = [item['content'] for item in self.index_data]
        metadatas = [{k: v for k, v in item.items() if k != 'content'} for item in self.index_data]
        self.vector_store = FAISS.from_texts(docs, embedding_model, metadatas=metadatas)

        # Create BM25 index
        tokenized_corpus = [doc.lower().split() for doc in self.raw_texts]
        self.bm25_index = BM25Okapi(tokenized_corpus)

        logger.info(f"Created JS API indexes with {self.vector_store.index.ntotal} methods")

        return self.vector_store

    def search(self, query: str, k: int = 10) -> List[Dict]:
        """Search JS API methods using hybrid retrieval"""
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
            doc_id = f"{doc.get('type', '')}-{doc.get('name', '')}"
            if doc_id not in seen_ids:
                seen_ids.add(doc_id)
                combined_docs.append(doc)

        return combined_docs[:k]
