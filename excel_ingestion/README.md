# Excel Ingestion Azure Function

## Overview 
This Azure Function provides an HTTP endpoint for processing Excel files stored in Azure Blob Storage. It converts tabular data into narrative text using Azure OpenAI's GPT-4o model. The processed data is then uploaded back to the Blob Storage container.

## Requirements
- Azure Functions Core Tools
- Azure CLI
- Python 3.10+
- Azure Storage Account
- Authentication with DefaultAzureCredential
- Azure Blob Storage Container Name

## Architecture

The function follows a modular architecture with the following components:

1. ExcelIngestionHandler: Main orchestrator class
2. BlobStorage: Handles Azure Blob Storage operations
3. ExcelProcessor: Manages Excel file processing and data transformation
4. ChunkProcessor: Handles parallel processing of data chunks
5. LLMManager: Manages LLM operations using Azure OpenAI

## API Specification

### Endpoint 

POST /api/excel-ingestion

### Request Body

```json
{
    "excel_blob_path": "path/to/excel/file.xlsx"
}
```

### Response Body

```json
{
    "status": "success",
    "message": "Excel file processed successfully",
    "output_path": "path/to/processed/file.txt" (in the specified container)
}
```

### Error code 

400: Bad Request - Invalid request body
500: Internal Server Error - Error processing Excel file

## Processing Flow

1. Request validation 
- Validate the request body
- Ensure the request body contains required fields

2. Excel file processing
- Download the Excel file from Blob Storage
- Process the Excel file (column renamming, remove unnamed columns)
- Split data into manageable chunks for LLM to process

3. LLM Processing 
- Processes data chunks in parallel 
- Applies system prompt for consistent analysis 
- Generates narrative text for each chunk 

4. Upload processed data 
- Uploads the processed data to the specified Blob Storage container
- Returns the path to the processed file 

## Authentication 
- Uses Microsoft Entra ID for authentication 
- Require RBAC permissinos for Azure Blob Storage and Azure OpenAI service

## Configuration Constants
- MAX_WORKERS: Maximum number of workers for parallel processing
- LLM_DEPLOYMENT_NAME: Name of the Azure OpenAI deployment to use
- CONTAINER_NAME: Name of the Azure Blob Storage container to use

## Usage Example

```python
import requests

url = "https://your-function-url/api/excel-ingestion"
payload = {
    "excel_blob_path": "reports/quarterly_data.xlsx"
}
headers = {
    "x-request-id": "unique-request-id"
}

response = requests.post(url, json=payload, headers=headers)
result = response.json()

```
