import numpy as np
import logging
import json
import os
from typing import Dict, List
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from google.oauth2.credentials import Credentials
from google.auth.transport.requests import Request
from google_auth_oauthlib.flow import InstalledAppFlow
from langchain_community.vectorstores import FAISS
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from rank_bm25 import BM25Okapi
from config import EMBEDDING_MODEL, GDRIVE_FILE_ID, GDRIVE_CREDENTIALS_PATH, JS_API_FALLBACK_FILE

logger = logging.getLogger(__name__)

class JsApiIndexer:
    """Handles indexing and retrieval of SL2 JavaScript API methods"""

    def __init__(self, embedding_model_name: str = EMBEDDING_MODEL, gdrive_file_id: str = GDRIVE_FILE_ID, credentials_path: str = GDRIVE_CREDENTIALS_PATH):
        """Initialize the JS API indexer"""
        self.embedding_model_name = embedding_model_name
        self.gdrive_file_id = gdrive_file_id  # Google Drive file ID instead of path
        self.credentials_path = credentials_path or 'credentials.json'
        self.token_path = 'token.json'
        self.scopes = ['https://www.googleapis.com/auth/drive.readonly']
        self.drive_service = None
        self.vector_store = None  # Add missing vector_store initialization
        self.index_data = []
        self.bm25_index = None
        self.raw_texts = []
        self.last_file_mtime = None
        self._setup_drive_service()

    def _setup_drive_service(self):
        """Setup Google Drive API service"""
        try:
            creds = None
            # Load existing token
            if os.path.exists(self.token_path):
                creds = Credentials.from_authorized_user_file(self.token_path, self.scopes)
            
            # If there are no (valid) credentials available, let the user log in
            if not creds or not creds.valid:
                if creds and creds.expired and creds.refresh_token:
                    creds.refresh(Request())
                else:
                    if not os.path.exists(self.credentials_path):
                        logger.error(f"Credentials file not found: {self.credentials_path}")
                        return
                    
                    flow = InstalledAppFlow.from_client_secrets_file(
                        self.credentials_path, self.scopes)
                    creds = flow.run_local_server(port=0)
                
                # Save the credentials for the next run
                with open(self.token_path, 'w') as token:
                    token.write(creds.to_json())

            # Build the Drive service
            self.drive_service = build('drive', 'v3', credentials=creds)
            logger.info("Google Drive API service initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to setup Google Drive API service: {e}")
            self.drive_service = None

    def _get_js_api_file_info(self) -> Dict:
        """Get JS API file info from Google Drive"""
        if not self.drive_service or not self.gdrive_file_id:
            return {}
        
        try:
            js_api_file_info = self.drive_service.files().get(
                fileId=self.gdrive_file_id,
                fields='id,name,modifiedTime,mimeType'
            ).execute()
            return js_api_file_info
        except HttpError as e:
            logger.error(f"Error getting JS API file info: {e}")
            return {}

    def _download_file_content(self) -> str:
        """Download file content from Google Drive"""
        if not self.drive_service or not self.gdrive_file_id:
            return None
        
        try:
            # Get file content
            request = self.drive_service.files().get_media(fileId=self.gdrive_file_id)
            file_content = request.execute()
            return file_content.decode('utf-8')
        except HttpError as e:
            logger.error(f"Error downloading file content: {e}")
            return None

    def _load_data_from_file(self) -> List[Dict]:
        """Load JS API data from Google Drive document"""
        try:
            if not self.drive_service:
                logger.warning("Google Drive service not available. Using fallback data.")
                return self._get_fallback_data()

            # Download file content
            file_content = self._download_file_content()
            if not file_content:
                logger.warning("Could not download file content. Using fallback data.")
                return self._get_fallback_data()

            # Parse JSON content
            data = json.loads(file_content)
            
            # Validate data structure
            if isinstance(data, list):
                for item in data:
                    required_fields = ['type', 'name', 'description', 'syntax', 'example', 'content']
                    if not all(field in item for field in required_fields):
                        logger.warning(f"Invalid data structure in Google Drive file")
                        return self._get_fallback_data()
                
                logger.info(f"Loaded {len(data)} JS API methods from Google Drive")
                return data
            else:
                logger.warning(f"Invalid data format in Google Drive file")
                return self._get_fallback_data()

        except json.JSONDecodeError as e:
            logger.error(f"JSON decode error in Google Drive file: {e}")
            return self._get_fallback_data()
        except Exception as e:
            logger.error(f"Error loading data from Google Drive: {e}")
            return self._get_fallback_data()

    def _get_fallback_data(self) -> List[Dict]:
        """Return fallback data from local JSON file if Google Drive file cannot be loaded"""
        fallback_file_path = JS_API_FALLBACK_FILE
        
        try:
            if os.path.exists(fallback_file_path):
                with open(fallback_file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # Validate data structure
                if isinstance(data, list):
                    for item in data:
                        required_fields = ['type', 'name', 'description', 'syntax', 'example', 'content']
                        if not all(field in item for field in required_fields):
                            logger.warning(f"Invalid data structure in {fallback_file_path}, using hardcoded fallback")
                            return self._get_hardcoded_fallback_data()
                    
                    logger.info(f"Loaded {len(data)} JS API methods from fallback file: {fallback_file_path}")
                    return data
                else:
                    logger.warning(f"Invalid data format in {fallback_file_path}, using hardcoded fallback")
                    return self._get_hardcoded_fallback_data()
            else:
                logger.warning(f"Fallback file not found: {fallback_file_path}, using hardcoded fallback")
                return self._get_hardcoded_fallback_data()
                
        except json.JSONDecodeError as e:
            logger.error(f"JSON decode error in {fallback_file_path}: {e}, using hardcoded fallback")
            return self._get_hardcoded_fallback_data()
        except Exception as e:
            logger.error(f"Error loading fallback data from {fallback_file_path}: {e}, using hardcoded fallback")
            return self._get_hardcoded_fallback_data()

    def _get_hardcoded_fallback_data(self) -> List[Dict]:
        """Return hardcoded fallback data as last resort"""
        return [
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

    def _file_has_changed(self) -> bool:
        """Check if the Google Drive JS API file has been modified"""
        try:
            if not self.drive_service or not self.gdrive_file_id:
                return False
            
            js_api_file_info = self._get_js_api_file_info()
            if not js_api_file_info:
                return False
            
            current_modified_time = js_api_file_info.get('modifiedTime')
            if not current_modified_time:
                return False
            
            # Convert to timestamp for comparison
            from datetime import datetime
            import dateutil.parser
            
            current_timestamp = dateutil.parser.parse(current_modified_time).timestamp()
            
            if self.last_file_mtime is None or current_timestamp > self.last_file_mtime:
                self.last_file_mtime = current_timestamp
                return True
            return False
            
        except Exception as e:
            logger.error(f"Error checking JS API file modification time: {e}")
            return False

    def extract_index_data(self) -> List[Dict]:
        """Extract JS API methods for indexing from Google Drive document"""
        # Check if we need to reload data
        if self._file_has_changed() or not self.index_data:
            logger.info("File changed or no data loaded, reloading JS API data")
            js_api_methods = self._load_data_from_file()
            self.index_data = js_api_methods
            self.raw_texts = [item['content'] for item in js_api_methods]
            
            # Reset indexes to force recreation
            self.vector_store = None
            self.bm25_index = None

        return self.index_data

    def create_indexes(self):
        """Create vector store and BM25 index for JS API methods"""
        # Always check for data changes before creating indexes
        self.extract_index_data()

        if self.vector_store is None or self.bm25_index is None:
            # Create embeddings using sentence-transformers
            embedding_model = SentenceTransformerEmbeddings(model_name=self.embedding_model_name)

            # Create FAISS index for JS API
            docs = [item['content'] for item in self.index_data]
            js_api_data = [{k: v for k, v in item.items() if k != 'content'} for item in self.index_data]
            self.vector_store = FAISS.from_texts(docs, embedding_model, metadatas=js_api_data)

            # Create BM25 index
            tokenized_corpus = [doc.lower().split() for doc in self.raw_texts]
            self.bm25_index = BM25Okapi(tokenized_corpus)

            logger.info(f"Created JS API indexes with {self.vector_store.index.ntotal} methods")

        return self.vector_store

    def search(self, query: str, k: int = 10) -> List[Dict]:
        """Search JS API methods using hybrid retrieval"""
        # Check for file changes before searching
        if self._file_has_changed():
            logger.info("File changed, recreating indexes")
            
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
