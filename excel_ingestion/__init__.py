import logging
import azure.functions as func
from azure.identity import DefaultAzureCredential
from shared.blob_storage import BlobStorage
from .ingestion import *

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [Ingestion-Webbackend] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


LLM_DEPLOYMENT_NAME = "Agent"
MAX_WORKERS = 5
CONTAINER_NAME = "documents"


""" 
Here is the sequence of steps: 
1. get the excel file from the blob storage 
2. read the excel file 
3. clean the excel file 
4. convert the excel file to markdown 
5. upload the markdown file to the blob storage 
"""

blob_storage = BlobStorage()
def main(req: func.HttpRequest) -> func.HttpResponse:
    logging.info('Python HTTP trigger function processed a request.')

    # request body should look like this:
    # {
    #     "excel_blob_path": "string",
    # }


    # Get the JSON data from the request body
    request_body = req.get_json()
    
    # get the local blob path the excel container 

    excel_blob_path = request_body.get('excel_blob_path')

    # initialize the excel processor 
    excel_processor = ExcelProcessor()

    #########################################
    # Pre Prcess the excel file (data cleaning)
    #########################################

    # download the excel file from the blob storage 
    df = excel_processor.download_and_read_excel_file(excel_blob_path)

    # rename the columns (databook)
    df = excel_processor.rename_columns(df)

    # remove the unnamed columns 
    df = excel_processor.remove_unnamed_columns(df)

    # get the date of the excel file 
    date = excel_processor.get_date(excel_blob_path)

    # vertically split the dataframe (cut in half, or more depends on the number of column)
    df_parts = excel_processor.split_data(df)

    # validate the splits 
    validate_splits(df_parts)

    #########################################
    # Process the excel file 
    #########################################

    # initialize the chunk processor 
    chunk_processor = ChunkProcessor(max_workers=MAX_WORKERS)

    # initialize the llm manager 
    llm_manager = LLMManager(deployment_name="Agent")

    # get the system prompt 
    system_prompt = llm_manager.get_prompt("excel_processing_system_prompt")

    # parallel llm chunk processing 
    md_output = chunk_processor.parallel_llm_chunk_processing(df_parts=df_parts, 
                                                    date=date, 
                                                    llm_manager=llm_manager, 
                                                    system_prompt=system_prompt)
    
    # upload the output to blob storage 
    blob_storage.container_client.upload_blob(
        name=f"Excel-Processed/{excel_blob_path.replace('.xlsx', '_processed.md')}", 
        data=md_output, 
        content_settings=ContentSettings('text/plain'), 
        overwrite=True)

    return func.HttpResponse(status_code=200, body="Excel file processed successfully")





    



# create 3 classes? 
""" 
1. Provide a blob link to the 
2. import the data clean it 
3. LLM Processing
4. Upload to Blob    
"""