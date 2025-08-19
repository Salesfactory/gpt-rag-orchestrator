"""
Report Registry

This module provides a registry system for different report generators.
Each report type has a unique key and associated generator function.
"""

import logging
from typing import Dict, Callable, Any, Optional
from abc import ABC, abstractmethod

class ReportGeneratorBase(ABC):
    """Base class for all report generators"""
    
    @abstractmethod
    async def generate(self, job_id: str, organization_id: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate a report and return metadata about the generated artifact.
        
        Args:
            job_id: Unique identifier for the report job
            organization_id: Organization requesting the report
            parameters: Report-specific parameters
            
        Returns:
            Dict containing:
            - blob_url: URL to the generated report artifact
            - file_name: Name of the generated file
            - file_size: Size of the file in bytes
            - content_type: MIME type of the generated file
            - metadata: Any additional metadata about the report
        """
        pass

class SampleReportGenerator(ReportGeneratorBase):
    """Sample report generator for demonstration purposes"""
    
    async def generate(self, job_id: str, organization_id: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Generate a sample report"""
        import json
        from datetime import datetime
        from azure.storage.blob import BlobServiceClient
        import os
        
        # Create sample report content
        report_content = {
            "report_type": "sample",
            "job_id": job_id,
            "organization_id": organization_id,
            "generated_at": datetime.utcnow().isoformat(),
            "parameters": parameters,
            "data": {
                "message": "This is a sample report",
                "status": "completed"
            }
        }
        
        # Convert to JSON
        json_content = json.dumps(report_content, indent=2)
        file_name = f"sample_report_{job_id}.json"
        
        # Upload to blob storage
        connection_string = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
        if not connection_string:
            raise ValueError("AZURE_STORAGE_CONNECTION_STRING environment variable not set")
            
        blob_service_client = BlobServiceClient.from_connection_string(connection_string)
        container_name = "reports"
        blob_name = f"{organization_id}/{file_name}"
        
        # Ensure container exists
        try:
            container_client = blob_service_client.get_container_client(container_name)
            container_client.get_container_properties()
        except Exception:
            container_client = blob_service_client.create_container(container_name)
            
        # Upload the report
        blob_client = blob_service_client.get_blob_client(
            container=container_name, 
            blob=blob_name
        )
        
        blob_client.upload_blob(
            json_content.encode('utf-8'), 
            overwrite=True,
            content_settings={
                'content_type': 'application/json'
            }
        )
        
        return {
            "blob_url": blob_client.url,
            "file_name": file_name,
            "file_size": len(json_content.encode('utf-8')),
            "content_type": "application/json",
            "metadata": {
                "records_processed": 1,
                "report_version": "1.0"
            }
        }

class ConversationAnalyticsGenerator(ReportGeneratorBase):
    """Generate analytics reports from conversation data"""
    
    async def generate(self, job_id: str, organization_id: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Generate conversation analytics report"""
        # This would implement actual conversation analytics logic
        # For now, return a placeholder
        raise NotImplementedError("Conversation analytics generator not yet implemented")

class UsageReportGenerator(ReportGeneratorBase):
    """Generate usage reports for organization"""
    
    async def generate(self, job_id: str, organization_id: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Generate usage report"""
        # This would implement actual usage reporting logic
        # For now, return a placeholder
        raise NotImplementedError("Usage report generator not yet implemented")

# Registry of available report generators
_REPORT_GENERATORS: Dict[str, ReportGeneratorBase] = {
    "sample": SampleReportGenerator(),
    "conversation_analytics": ConversationAnalyticsGenerator(),
    "usage_report": UsageReportGenerator(),
}

def get_generator(report_key: str) -> Optional[ReportGeneratorBase]:
    """
    Get a report generator by its key.
    
    Args:
        report_key: The unique key identifying the report type
        
    Returns:
        ReportGeneratorBase instance or None if not found
    """
    generator = _REPORT_GENERATORS.get(report_key)
    if generator is None:
        logging.warning(f"No generator found for report key: {report_key}")
    return generator

def register_generator(report_key: str, generator: ReportGeneratorBase) -> None:
    """
    Register a new report generator.
    
    Args:
        report_key: Unique key for the report type
        generator: ReportGeneratorBase instance
    """
    if not isinstance(generator, ReportGeneratorBase):
        raise ValueError("Generator must inherit from ReportGeneratorBase")
        
    _REPORT_GENERATORS[report_key] = generator
    logging.info(f"Registered report generator: {report_key}")

def list_available_generators() -> Dict[str, str]:
    """
    List all available report generators.
    
    Returns:
        Dict mapping report keys to generator class names
    """
    return {key: generator.__class__.__name__ for key, generator in _REPORT_GENERATORS.items()}
