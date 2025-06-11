"""
Example usage of RagManager with Google Drive integration
"""

from rag_manager import RagManager
from config import GDRIVE_FILE_ID, GDRIVE_CREDENTIALS_PATH

def main():
    # Initialize RAG Manager with Google Drive configuration
    rag_manager = RagManager(
        metadata_path='metadata.json',
        gdrive_file_id=GDRIVE_FILE_ID,  # Your Google Drive file ID
        credentials_path=GDRIVE_CREDENTIALS_PATH  # Path to credentials.json
    )
    
    # Initialize all indexers
    try:
        rag_manager.initialize()
        print("RAG Manager initialized successfully")
    except Exception as e:
        print(f"Error initializing RAG Manager: {e}")
        return
    
    # Example search
    query = "How to get model data using JavaScript API?"
    results = rag_manager.hybrid_search(query, k=10)
    
    print(f"\nSearch results for: '{query}'")
    print(f"Found {len(results)} results:")
    
    for i, result in enumerate(results, 1):
        print(f"\n{i}. {result.get('name', 'Unknown')}")
        print(f"   Type: {result.get('type', 'Unknown')}")
        if result.get('description'):
            print(f"   Description: {result['description'][:100]}...")

if __name__ == "__main__":
    main()
