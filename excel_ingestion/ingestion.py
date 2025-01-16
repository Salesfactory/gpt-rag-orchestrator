# Standard library imports
import copy
import io
import json
import os
import sys
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

# Third party imports
import numpy as np
import pandas as pd
from tqdm import tqdm
from azure.storage.blob import ContentSettings

# Add parent directory to Python path
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

# Local imports
from shared.blob_storage import BlobStorage, BlobDownloadError
from shared.llm_config import LLMManager

# Maximum number of concurrent threads for parallel processing of Excel files
MAX_WORKERS = 3
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [ExcelIngestion] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

############################################
# ExcelProcessor Class
############################################

""" 
This class is responsible for downloading and reading excel files from blob storage, 
renaming columns, splitting the data, and removing unnamed columns.
"""

class ExcelProcessor:
    def __init__(self):
        self.blob_storage = BlobStorage()

    def download_and_read_excel_file(self, file_path: str) -> pd.DataFrame:
        """ 
        Download an excel file from blob storage and read it into a pandas DataFrame.

        Args:
            file_path (str): The path to the excel file in blob storage. (e.g. "Databook_Nov8th2024.xlsx")

        Returns:
            pd.DataFrame: The excel file as a pandas DataFrame.
        """
        try:
            blob_client = self.blob_storage.container_client.get_blob_client(blob=file_path)
            downloaded_blob = blob_client.download_blob()
            blob_data = downloaded_blob.readall()
        except Exception as e:
            raise BlobDownloadError(f"Error downloading and reading excel file: {str(e)}")
        try: 
            df = pd.read_excel(io.BytesIO(blob_data), skiprows=1)
            return df
        except Exception as e:
            raise ValueError(f"Error reading excel file: {str(e)}")

    def rename_columns(self, df):
        rename_map = {
            'Female': 'Gender_Female',
            'Male': 'Gender_Male',
            'Gen Z': 'Generation_GenZ',
            'Millennial': 'Generation_Millennial',
            'Gen X': 'Generation_GenX',
            'Boomer': 'Generation_Boomer',
            'Affluent & Educated': 'Segmentation_Affluent_and_Educated',
            'Aspiring Singles': 'Segmentation_Aspiring_Singles',
            'Cautious & Accepting': 'Segmentation_Cautious_and_Accepting',
            'Stable Strategists': 'Segmentation_Stable_Strategist',
            'Sunsetting Suburbanites': 'Segmentation_Sunsetting_Suburbanites',
            'Republican': 'Political_Affiliation_Republican',
            'Democratic': 'Political_Affiliation_Democratic',
            'Independent/Unaffiliated': 'Political_Affiliation_Independent',
            'Other': 'Political_Affiliation_Other',
            'Unnamed: 0': 'Section'
        }
        
        # Only rename columns that exist in the DataFrame
        existing_rename_map = {k: v for k, v in rename_map.items() if k in df.columns}
        df = df.rename(columns=existing_rename_map)
        
        return df
    
    def remove_first_row(self, df):
        # Use first row as column names and drop that row
        df.columns = df.iloc[0]
        return df.iloc[1:]

    def split_data(self, df: pd.DataFrame, 
                   row_name_col: str = 'Section') -> tuple:
        """
        Splits a DataFrame into multiple parts based on number of columns, ensuring the specified 
        row_name_col column is present in all parts.
            
        Parameters:
        df: pd.DataFrame
            The DataFrame to split.
        row_name_col: str, default 'Section'
            The column name to ensure is present in all parts.
        
        Returns:
        tuple: Contains 2-4 DataFrames depending on number of columns, or message if too few/many columns
        """
        if row_name_col not in df.columns:
            raise ValueError(f"Column '{row_name_col}' not found in DataFrame.")
        
        num_cols = len(df.columns)
        
        if num_cols > 32:
            raise ValueError("Too many columns (>32)")
        elif num_cols <= 8:
            return (df.copy(),)
        
        # Determine number of splits
        if num_cols > 24:
            num_splits = 4
        elif num_cols > 18:
            num_splits = 3
        elif num_cols > 8:
            num_splits = 2
        elif num_cols > 0:
            num_splits = 1
            
        # Calculate split indices
        split_size = num_cols // num_splits
        split_indices = [i * split_size for i in range(1, num_splits)]
        if num_cols % num_splits != 0:
            split_indices[-1] += num_cols % num_splits
            
        # Split dataframe
        dfs = []
        start_idx = 0
        for end_idx in split_indices:
            df_part = df.iloc[:, start_idx:end_idx].copy()
            if row_name_col not in df_part.columns:
                df_part.insert(0, row_name_col, df[row_name_col])
            dfs.append(df_part)
            start_idx = end_idx
            
        # Handle last part
        df_part = df.iloc[:, start_idx:].copy()
        if row_name_col not in df_part.columns:
            df_part.insert(0, row_name_col, df[row_name_col])
        dfs.append(df_part)
        
        return tuple(dfs)

    def remove_unnamed_columns(self, df):
        return df.loc[:, ~df.columns.str.startswith('Unnamed')]

    def get_date(self, file_path):
        """
        Extract the date from a databook filename.
        
        Args:
            file_path (str): Path to databook file in format 'Databook_MonthDayYear.xlsx'
        
        Returns:
            str: Date string (e.g., 'Nov8th2024')
        """
        try:
            # Remove file extension and split by underscore
            parts = file_path.replace('.xlsx', '').split('_')
            if len(parts) < 2:
                raise ValueError("Invalid filename format")
            
            # Return the second part which contains the date
            return parts[1]
            
        except Exception as e:
            raise ValueError(f"Could not extract date from filename: {str(e)}")
    

############################################
# Chunking and LLM Processing
############################################

def validate_splits(df_parts: list):
    """
    Validate that the splits are valid.
    """
    if len(df_parts) == 0:
        raise ValueError("No splits found")
    elif len(df_parts) > 4:
        raise ValueError("Too many splits found")
    else: 
        logger.info(f"Valid splits found: {len(df_parts)}")
    return df_parts


class ChunkProcessor:
    def __init__(self, max_workers=3):
        self.max_workers = max_workers
        self.md_output = []

    def chunk_dataframe(self, df: pd.DataFrame, chunk_size: int = 10) -> list:
        """
        Splits a DataFrame into smaller chunks of specified size.
        
        Parameters:
        df: pd.DataFrame
            The DataFrame to split into chunks
        chunk_size: int, default 10
            The number of rows in each chunk
            
        Returns:
        list: List of DataFrame chunks
        """
        # Calculate number of chunks
        n_chunks = len(df) // chunk_size + (1 if len(df) % chunk_size != 0 else 0)
        
        # Split into chunks
        return [df.iloc[i*chunk_size:(i+1)*chunk_size].copy() 
                for i in range(n_chunks)]

    def llm_process_chunk(self, chunk, date, llm_manager, system_prompt):
        """Helper function to process individual chunks"""
        md_chunk = chunk.to_markdown()
        user_message = f"""Here is the created date of data: {date}

        Here is the data table: 
        
        ```
        {md_chunk}
        ```
        """
        return llm_manager.get_response("Agent", use_langchain=False, 
                                      custom_prompt=system_prompt, 
                                      user_message=user_message)

    def parallel_llm_chunk_processing(self, df_parts, date, llm_manager, system_prompt):
        """Main method to process all dataframe parts"""
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = []
            
            # Submit all chunks to the executor
            for part in df_parts:
                chunks = self.chunk_dataframe(part)
                for chunk in chunks[:3]:  # Still limiting to first 3 chunks per part
                    future = executor.submit(self.llm_process_chunk, chunk, date, llm_manager, system_prompt)
                    futures.append(future)
            
            # Process results as they complete with progress bar
            for future in tqdm(as_completed(futures), total=len(futures), desc="Processing chunks"):
                try:
                    result = future.result()
                    self.md_output.append(result)
                except Exception as e:
                    print(f"Error processing chunk: {str(e)}")

        # Join results with separator
        return "\n----------------------------------------------------------------------------\n".join(self.md_output)

if __name__ == "__main__":
    file_path = "Databook_Nov8th2024.xlsx"
    excel_processor = ExcelProcessor()
    df = excel_processor.download_and_read_excel_file(file_path)
    df = excel_processor.rename_columns(df)
    df = excel_processor.remove_unnamed_columns(df)
    date = excel_processor.get_date(file_path)
    df_parts = excel_processor.split_data(df)
    validate_splits(df_parts)
    chunk_processor = ChunkProcessor(max_workers=MAX_WORKERS)
    llm_manager = LLMManager(deployment_name="Agent")
    system_prompt = llm_manager.get_prompt("excel_processing_system_prompt")

    # get dimensions of the dataframe
    df_dimensions = df.shape
    logger.info(f"Dataframe dimensions: {df_dimensions}")
    md_output = chunk_processor.parallel_llm_chunk_processing(df_parts=df_parts, 
                                                    date=date, 
                                                    llm_manager=llm_manager, 
                                                    system_prompt=system_prompt)
    
    # upload to blob storage
    blob_storage = BlobStorage()
    blob_storage.container_client.upload_blob(
        name=f"Excel-Processed/{file_path.replace('.xlsx', '_processed.md')}", 
        data=md_output, 
        content_settings=ContentSettings('text/plain'), 
        overwrite=True)


