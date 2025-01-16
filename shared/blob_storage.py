from azure.identity import DefaultAzureCredential
from azure.storage.blob import BlobServiceClient, ContentSettings
import os
import pandas as pd
import io
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [BlobStorage] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


############################################
# Custom Blob Storage Class
############################################


class BlobStorageError(Exception):
    """Exception raised for errors in the BlobStorage class."""
    def __init__(self, message: str):
        self.message = message
        super().__init__(self.message)
        
class BlobDownloadError(BlobStorageError):
    """Exception raised for errors in the BlobStorage class."""
    def __init__(self, message: str):
        self.message = message
        super().__init__(self.message)

class BlobInitError(BlobStorageError):
    """Exception raised for errors in the BlobStorage class."""
    def __init__(self, message: str):
        self.message = message
        super().__init__(self.message)

class BlobStorage:
    def __init__(self, container_name: str = "documents"):
        credential = DefaultAzureCredential()
        BLOB_ACCOUNT_URL = f"https://{os.getenv('STORAGE_ACCOUNT')}.blob.core.windows.net"
        try:
            self.blob_service_client = BlobServiceClient(account_url=BLOB_ACCOUNT_URL, credential=credential)
            self.container_client = self.blob_service_client.get_container_client(container_name)
        except Exception as e:
            raise BlobInitError(f"Error initializing BlobStorage: {e}")

    def list_excel_files(self):
        # list all excel files in the container 
        try: 
            blob_list = self.container_client.list_blobs()
            return [blob.name for blob in blob_list if blob.name.endswith('.xlsx')]
        except Exception as e:
            logger.error(f"Error listing excel files: {e}")
            return []

if __name__ == "__main__":
    blob_storage = BlobStorage()
    
    # download files 
    # blob_client = blob_storage.container_client.get_blob_client(blob="Databook_Nov8th2024.xlsx")
    # downloaded_blob = blob_client.download_blob()
    
    # # Convert blob data to bytes and read with pandas
    # blob_data = downloaded_blob.readall()
    # df = pd.read_excel(io.BytesIO(blob_data))
    # print(df.head())
    # list all excel files in the container 
    print(blob_storage.list_excel_files())


