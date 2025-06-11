"""
Setup script for Google Drive API integration
1. Go to https://console.cloud.google.com/
2. Create a new project or select existing one
3. Enable Google Drive API
4. Create credentials (OAuth 2.0 Client ID for desktop application)
5. Download credentials.json file
6. Run this script to get the file ID from a Google Drive file URL
"""

import re

def extract_file_id_from_url(gdrive_url: str) -> str:
    """Extract file ID from Google Drive URL"""
    patterns = [
        r'/file/d/([a-zA-Z0-9-_]+)',
        r'id=([a-zA-Z0-9-_]+)',
        r'/d/([a-zA-Z0-9-_]+)'
    ]
    
    for pattern in patterns:
        match = re.search(pattern, gdrive_url)
        if match:
            return match.group(1)
    
    return None

def main():
    print("Google Drive API Setup Helper")
    print("=" * 40)
    
    # Get file URL from user
    gdrive_url = input("Enter your Google Drive file URL: ").strip()
    
    # Extract file ID
    file_id = extract_file_id_from_url(gdrive_url)
    
    if file_id:
        print(f"\nExtracted File ID: {file_id}")
        print(f"\nUse this file ID in your JsApiIndexer:")
        print(f"indexer = JsApiIndexer(gdrive_file_id='{file_id}')")
    else:
        print("Could not extract file ID from the URL")
        print("Make sure the URL is a valid Google Drive file URL")

if __name__ == "__main__":
    main()
