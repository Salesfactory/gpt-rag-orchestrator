import os
import logging
import traceback
from typing import Dict, Any
from http import HTTPStatus

import azure.functions as func
from azure.core.exceptions import AzureError
from shared.blob_storage import BlobStorage
from azure.identity import DefaultAzureCredential
from azure.storage.blob import ContentSettings, BlobClient
from .ingestion import ExcelProcessor, ChunkProcessor, LLMManager
from .schemas import ExcelIngestionRequest, ExcelIngestionResponse
from .exceptions import ValidationError, ProcessingError

# Define logger name constant
LOGGER_NAME = "[ExcelIngestion-Function]"

logging.basicConfig(
    level=logging.INFO,
    format=f'%(asctime)s {LOGGER_NAME} %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

MAX_WORKERS = 10
LLM_DEPLOYMENT_NAME = "Agent"
CONTAINER_NAME = "documents"
BLOB_ACCOUNT_URL = f"https://{os.getenv('STORAGE_ACCOUNT')}.blob.core.windows.net"

class ExcelIngestionHandler:
    def __init__(self):
        logger.info(f"{LOGGER_NAME} Initializing ExcelIngestionHandler")
        self.blob_storage = BlobStorage(container_name=CONTAINER_NAME)
        self.excel_processor = ExcelProcessor()
        self.chunk_processor = ChunkProcessor(max_workers=MAX_WORKERS)
        self.llm_manager = LLMManager(deployment_name=LLM_DEPLOYMENT_NAME)
        logger.info(f"{LOGGER_NAME} ExcelIngestionHandler initialized successfully")

    def process_excel(self, excel_blob_path: str) -> str:
        """Process excel file and return markdown output path."""
        logger.info(f"{LOGGER_NAME} Starting excel processing for file: {excel_blob_path}")

        logger.info(f"{LOGGER_NAME} Validating excel blob path")
        self._validate_excel_blob_path(excel_blob_path)

        try:
            # Download and preprocess
            logger.info(f"{LOGGER_NAME} Downloading and preprocessing excel file")
            df = self.excel_processor.download_and_read_excel_file(excel_blob_path)
            df = self._preprocess_dataframe(df)
            
            # Process with LLM
            logger.info(f"{LOGGER_NAME} Processing with LLM")
            date = self.excel_processor.get_date(excel_blob_path)
            df_parts = self.excel_processor.split_data(df)
            system_prompt = self.llm_manager.get_prompt("excel_processing_system_prompt")
            
            md_output = self.chunk_processor.parallel_llm_chunk_processing(
                df_parts=df_parts,
                date=date,
                llm_manager=self.llm_manager,
                system_prompt=system_prompt
            )

            # Upload result
            output_path = f"Excel-Processed/{excel_blob_path.replace('.xlsx', '_processed.md')}"
            logger.info(f"{LOGGER_NAME} Uploading processed markdown to: {output_path}")
            self._upload_markdown(output_path, md_output)
            
            logger.info(f"{LOGGER_NAME} Excel ingested and uploaded completed successfully")
            return output_path

        except Exception as e:
            logger.error(f"{LOGGER_NAME} Error processing excel file: {str(e)}")
            raise ProcessingError(f"Failed to process excel file: {str(e)}")
    
    def _validate_excel_blob_path(self, excel_blob_path: str):
        """Validate excel blob path."""
        if not excel_blob_path.endswith('.xlsx'):
            raise ValidationError("Excel file must have a .xlsx extension")
        
        if not BlobClient(account_url=BLOB_ACCOUNT_URL, credential=DefaultAzureCredential(), container_name=CONTAINER_NAME, blob_name=excel_blob_path).exists():
            raise ValidationError(f"Excel file not found in blob storage: {excel_blob_path}")
        
    def _preprocess_dataframe(self, df):
        """Handle all dataframe preprocessing steps."""
        logger.debug(f"{LOGGER_NAME} Preprocessing dataframe")
        df = self.excel_processor.rename_columns(df)
        df = self.excel_processor.remove_unnamed_columns(df)
        return df

    def _upload_markdown(self, path: str, content: str) -> None:
        """Upload markdown content to blob storage."""
        try:
            logger.info(f"{LOGGER_NAME} Uploading markdown to blob storage: {path}")
            self.blob_storage.container_client.upload_blob(
                name=path,
                data=content,
                content_settings=ContentSettings('text/plain'),
                overwrite=True
            )
            logger.info(f"{LOGGER_NAME} Markdown uploaded successfully")
        except AzureError as e:
            logger.error(f"{LOGGER_NAME} Failed to upload markdown: {str(e)}")
            raise ProcessingError(f"Failed to upload markdown: {str(e)}")

def main(req: func.HttpRequest) -> func.HttpResponse:
    """Azure Function endpoint for excel ingestion."""
    request_id = req.headers.get('x-request-id', 'unknown')
    logger.info(f"{LOGGER_NAME} Processing excel ingestion request {request_id}")

    try:
        # Validate request
        if not req.get_json():
            logger.warning(f"{LOGGER_NAME} Request body is missing for request {request_id}")
            return func.HttpResponse(
                body='Request body is required',
                status_code=HTTPStatus.BAD_REQUEST
            )

        request_data = ExcelIngestionRequest(**req.get_json())
        
        # Process request
        logger.info(f"{LOGGER_NAME} Creating handler and processing excel file")
        handler = ExcelIngestionHandler()
        output_path = handler.process_excel(request_data.excel_blob_path)
        
        # Return response
        response = ExcelIngestionResponse(
            status="success",
            message="Excel file processed successfully",
            output_path=output_path
        )
        
        logger.info(f"{LOGGER_NAME} Request {request_id} processed successfully")
        return func.HttpResponse(
            body=response.model_dump_json(),
            status_code=HTTPStatus.OK,
            mimetype="application/json"
        )

    except ValidationError as e:
        logger.warning(f"{LOGGER_NAME} Validation error for request {request_id}: {str(e)}")
        return func.HttpResponse(
            body=str(e),
            status_code=HTTPStatus.BAD_REQUEST
        )
    
    except ProcessingError as e:
        logger.error(f"{LOGGER_NAME} Processing error for request {request_id}: {str(e)}")
        return func.HttpResponse(
            body=str(e),
            status_code=HTTPStatus.INTERNAL_SERVER_ERROR
        )
    
    except Exception as e:
        logger.error(f"{LOGGER_NAME} Unexpected error for request {request_id}: {str(e)}\n{traceback.format_exc()}")
        return func.HttpResponse(
            body=f"An unexpected error occurred: {str(e)}",
            status_code=HTTPStatus.INTERNAL_SERVER_ERROR
        )
