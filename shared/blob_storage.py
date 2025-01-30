from azure.identity import DefaultAzureCredential
from azure.storage.blob import BlobServiceClient
import os
import pandas as pd
import io
import logging

# Define logger name constant
LOGGER_NAME = "[BlobStorage]"

logging.basicConfig(
    level=logging.INFO,
    format=f'%(asctime)s {LOGGER_NAME} %(message)s',
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
        logger.info(f"{LOGGER_NAME} Initializing BlobStorage with container: {container_name}")
        credential = DefaultAzureCredential()
        BLOB_ACCOUNT_URL = f"https://{os.getenv('STORAGE_ACCOUNT')}.blob.core.windows.net"
        try:
            self.blob_service_client = BlobServiceClient(account_url=BLOB_ACCOUNT_URL, credential=credential)
            self.container_client = self.blob_service_client.get_container_client(container_name)
            # self.blob_client = BlobClient(account_url=BLOB_ACCOUNT_URL, credential=credential, container_name=container_name, blob_name=blob_name)
            logger.info(f"{LOGGER_NAME} Successfully initialized BlobStorage")
        except Exception as e:
            logger.error(f"{LOGGER_NAME} Failed to initialize BlobStorage: {str(e)}")
            raise BlobInitError(f"Error initializing BlobStorage: {e}")

    def list_excel_files(self):
        logger.info(f"{LOGGER_NAME} Listing Excel files in container")
        try: 
            blob_list = self.container_client.list_blobs()
            excel_files = [blob.name for blob in blob_list if blob.name.endswith('.xlsx')]
            logger.info(f"{LOGGER_NAME} Found {len(excel_files)} Excel files")
            return excel_files
        except Exception as e:
            logger.error(f"{LOGGER_NAME} Error listing Excel files: {str(e)}")
            return []
