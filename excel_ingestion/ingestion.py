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

# Define logger name constant
LOGGER_NAME = "[ExcelIngestion-Backend]"

logging.basicConfig(
    level=logging.INFO,
    format=f'%(asctime)s {LOGGER_NAME} %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

############################################
# ExcelProcessor Class
############################################

class ExcelProcessor:
    """
    A class for processing internal databook Excel files.
    
    This class handles downloading Excel files from Azure Blob Storage, data cleaning,
    column renaming, and data splitting operations. 

    Attributes:
        blob_storage (BlobStorage): An instance of BlobStorage for Azure Blob operations.
    """

    def __init__(self):
        """Initialize the ExcelProcessor with a BlobStorage connection to download."""
        logger.info(f"{LOGGER_NAME} Initializing ExcelProcessor")
        self.blob_storage = BlobStorage()

    def download_and_read_excel_file(self, file_path: str) -> pd.DataFrame:
        """ 
        Download and read an Excel file from Azure Blob Storage.

        This method handles both the download operation from Blob Storage and the
        conversion to a pandas DataFrame. It skips the first row during reading
        as it typically contains header information.

        Args:
            file_path (str): The path to the Excel file in blob storage (e.g., "Databook_Nov8th2024.xlsx")

        Returns:
            pd.DataFrame: The contents of the Excel file as a pandas DataFrame.

        Raises:
            BlobDownloadError: If there's an error downloading the file from Blob Storage.
            ValueError: If there's an error reading the Excel file into a DataFrame.
        """
        logger.info(f"{LOGGER_NAME} Downloading and reading excel file: {file_path}")
        try:
            blob_client = self.blob_storage.container_client.get_blob_client(blob=file_path)
            downloaded_blob = blob_client.download_blob()
            blob_data = downloaded_blob.readall()
        except Exception as e:
            logger.error(f"{LOGGER_NAME} Error downloading excel file: {str(e)}")
            raise BlobDownloadError(f"Error downloading and reading excel file: {str(e)}")
        try: 
            df = pd.read_excel(io.BytesIO(blob_data), skiprows=1)
            logger.info(f"{LOGGER_NAME} Successfully read excel file")
            return df
        except Exception as e:
            logger.error(f"{LOGGER_NAME} Error reading excel file: {str(e)}")
            raise ValueError(f"Error reading excel file: {str(e)}")

    def rename_columns(self, df):
        """
        Rename DataFrame columns to standardized format. Currently it's designed for databook excel files

        Transforms column names to be more descriptive and machine-friendly by adding
        prefixes for different categories (Gender_, Generation_, Segmentation_, etc.)
        and replacing spaces with underscores.

        Args:
            df (pd.DataFrame): DataFrame with original column names.

        Returns:
            pd.DataFrame: DataFrame with standardized column names.
        """
        logger.info(f"{LOGGER_NAME} Renaming columns")
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
        logger.info(f"{LOGGER_NAME} Successfully renamed columns")
        
        return df
    
    def remove_first_row(self, df):
        """
        Remove the first row of data and use it as column headers.

        Args:
            df (pd.DataFrame): Input DataFrame.

        Returns:
            pd.DataFrame: DataFrame with first row removed and used as column headers.
        """
        logger.info(f"{LOGGER_NAME} Removing first row")
        df.columns = df.iloc[0]
        return df.iloc[1:]

    def split_data(self, df: pd.DataFrame, 
                   row_name_col: str = 'Section') -> tuple:
        """
        Split a DataFrame into multiple parts based on number of columns.

        This method intelligently splits wide DataFrames into manageable chunks while
        ensuring the specified row name column is present in all resulting parts.
        The number of splits is determined by the total number of columns:
        - 1-8 columns: No split
        - 9-18 columns: 2 parts
        - 19-24 columns: 3 parts
        - 25-32 columns: 4 parts
            
        Args:
            df (pd.DataFrame): The DataFrame to split.
            row_name_col (str, optional): Column name to preserve in all splits. Defaults to 'Section'.
        
        Returns:
            tuple: Contains 1-4 DataFrames depending on the number of columns.

        Raises:
            ValueError: If row_name_col is not found or if DataFrame has >32 columns.
        """
        logger.info(f"{LOGGER_NAME} Splitting data")
        if row_name_col not in df.columns:
            logger.error(f"{LOGGER_NAME} Column '{row_name_col}' not found in DataFrame")
            raise ValueError(f"Column '{row_name_col}' not found in DataFrame.")
        
        num_cols = len(df.columns)
        
        if num_cols > 32:
            logger.error(f"{LOGGER_NAME} Too many columns (>32)")
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
        
        logger.info(f"{LOGGER_NAME} Successfully split data into {len(dfs)} parts")
        return tuple(dfs)

    def remove_unnamed_columns(self, df):
        """
        Remove columns whose names start with 'Unnamed'.

        Args:
            df (pd.DataFrame): Input DataFrame.

        Returns:
            pd.DataFrame: DataFrame with unnamed columns removed.
        """
        logger.info(f"{LOGGER_NAME} Removing unnamed columns")
        return df.loc[:, ~df.columns.str.startswith('Unnamed')]

    def get_date(self, file_path):
        """
        Extract the date from a databook filename.
        
        Parses filenames in the format 'Databook_MonthDayYear.xlsx' to extract
        the date portion.
        
        Args:
            file_path (str): Path to databook file (e.g., 'Databook_Nov8th2024.xlsx')
        
        Returns:
            str: Extracted date string (e.g., 'Nov8th2024')
        
        Raises:
            ValueError: If the filename format is invalid or date extraction fails.
        """
        logger.info(f"{LOGGER_NAME} Extracting date from filename: {file_path}")
        try:
            # Remove file extension and split by underscore
            parts = file_path.replace('.xlsx', '').split('_')
            if len(parts) < 2:
                logger.error(f"{LOGGER_NAME} Invalid filename format")
                raise ValueError("Invalid filename format")
            
            # Return the second part which contains the date
            logger.info(f"{LOGGER_NAME} Successfully extracted date: {parts[1]}")
            return parts[1]
            
        except Exception as e:
            logger.error(f"{LOGGER_NAME} Could not extract date from filename: {str(e)}")
            raise ValueError(f"Could not extract date from filename: {str(e)}")
############################################

def validate_splits(df_parts: list):
    """
    Validate that the splits are valid.
    """
    logger.info(f"{LOGGER_NAME} Validating splits")
    if len(df_parts) == 0:
        logger.error(f"{LOGGER_NAME} No splits found")
        raise ValueError("No splits found")
    elif len(df_parts) > 4:
        logger.error(f"{LOGGER_NAME} Too many splits found")
        raise ValueError("Too many splits found")
    else: 
        logger.info(f"{LOGGER_NAME} Valid splits found: {len(df_parts)}")
    return df_parts


class ChunkProcessor:
    def __init__(self, max_workers=3):
        logger.info(f"{LOGGER_NAME} Initializing ChunkProcessor with {max_workers} workers")
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
        logger.info(f"{LOGGER_NAME} Splitting DataFrame into chunks of size {chunk_size}")
        # Calculate number of chunks
        n_chunks = len(df) // chunk_size + (1 if len(df) % chunk_size != 0 else 0)
        
        # Split into chunks
        chunks = [df.iloc[i*chunk_size:(i+1)*chunk_size].copy() 
                for i in range(n_chunks)]
        logger.info(f"{LOGGER_NAME} Created {len(chunks)} chunks")
        return chunks

    def llm_process_chunk(self, chunk, date, llm_manager, system_prompt):
        """Helper function to process individual chunks"""
        logger.debug(f"{LOGGER_NAME} Processing chunk with LLM")
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
        logger.info(f"{LOGGER_NAME} Starting parallel LLM chunk processing")
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = []
            
            # Submit all chunks to the executor
            for part in df_parts:
                chunks = self.chunk_dataframe(part)
                for chunk in chunks:
                    future = executor.submit(self.llm_process_chunk, chunk, date, llm_manager, system_prompt)
                    futures.append(future)
            
            # Process results as they complete with progress bar
            for future in tqdm(as_completed(futures), total=len(futures), desc="Processing chunks"):
                try:
                    result = future.result()
                    self.md_output.append(result)
                except Exception as e:
                    logger.error(f"{LOGGER_NAME} Error processing chunk: {str(e)}")
                    print(f"Error processing chunk: {str(e)}")

        logger.info(f"{LOGGER_NAME} Completed parallel LLM chunk processing")
        # Join results with separator
        return "\n----------------------------------------------------------------------------\n".join(self.md_output)
