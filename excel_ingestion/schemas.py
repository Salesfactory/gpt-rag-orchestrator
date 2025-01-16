from pydantic import BaseModel, Field

class ExcelIngestionRequest(BaseModel):
    excel_blob_path: str = Field(..., description="Path to the excel file in blob storage")

class ExcelIngestionResponse(BaseModel):
    status: str
    message: str
    output_path: str