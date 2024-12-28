import json
import os
from azure.cosmos import CosmosClient, PartitionKey, exceptions, ThroughputProperties
import uuid
from datetime import datetime
from dotenv import load_dotenv
import logging

logging.basicConfig(
    level = logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class CosmosDBLoader:
    def __init__(self, container_name: str):
        load_dotenv()  # Load environment variables from .env file
        self.db_uri = os.getenv('AZURE_COSMOS_ENDPOINT')
        self.credential = os.getenv('AZURE_COSMOS_KEY')
        self.database_name = os.getenv('AZURE_DB_NAME')
        self.container_name = container_name or os.getenv('AZURE_COSMOS_CONTAINER_NAME')

        if not all([self.db_uri, self.credential, self.database_name, self.container_name]):
            raise ValueError("Missing required environment variables for Cosmos DB connection")

        self.client = CosmosClient(url=self.db_uri,
                                   credential= self.credential, 
                                   consistency_level="Session")
        self.database = self.client.get_database_client(self.database_name)
        self.container = self.database.get_container_client(self.container_name)

    def create_container(self):
        try: 
            self.container = self.database.create_container(
                id=self.container_name,
                partition_key=PartitionKey(path =["/companyId", "/reportType"], kind="MultiHash"),
                analytical_storage_ttl= -1,
                offer_throughput= 400, 
            )
            logger.info(f"Container {self.container_name} created successfully")
        except exceptions.CosmosResourceExistsError:
            logger.info(f"Container {self.container_name} already exists")
            self.container = self.database.get_container_client(self.container_name)
        except exceptions.CosmosHttpResponseError:
            raise 


    def upload_data(self, data_file_path: str) -> None:
        """
        Upload data from a JSON file to Cosmos DB
        """
        try:
            # Read the JSON file
            with open(data_file_path, 'r') as f:
                data = json.load(f)

            # Upload each schedule to Cosmos DB
            for item in data:
                item['id'] = str(uuid.uuid4())
                item['lastRun'] = datetime.now().isoformat()

                self.container.create_item(item)
            
            logger.info(f"Successfully uploaded {len(data)} items to Cosmos DB")
        except Exception as e:
            logger.error(f"Error uploading data to Cosmos DB: {str(e)}")
    
    def delete_data(self, company_id: str = None, report_type: str = None) -> None:
        try:
            if company_id and report_type:
                for item in self.container.query_items(
                    query=f"SELECT * FROM c WHERE c.companyId = '{company_id}' AND c.reportType = '{report_type}'",
                    enable_cross_partition_query=True
                ):
                    self.container.delete_item(item, partition_key=[item['companyId'], item['reportType']])
                logger.info(f"Successfully deleted {company_id} {report_type} from Cosmos DB")
                return True
            elif company_id:
                for item in self.container.query_items(
                    query=f"SELECT * FROM c WHERE c.companyId = '{company_id}'",
                    enable_cross_partition_query=True
                ):
                    self.container.delete_item(item, partition_key= [item['companyId'], item['reportType']])
                logger.info(f"Successfully deleted {company_id} from Cosmos DB")
                return True
            elif report_type:
                for item in self.container.query_items(
                    query=f"SELECT * FROM c WHERE c.reportType = '{report_type}'",
                    enable_cross_partition_query=True
                ):
                    self.container.delete_item(item, partition_key= [item['companyId'], item['reportType']])
                logger.info(f"Successfully deleted {report_type} from Cosmos DB")
                return True
        except Exception as e:
            logger.error(f"Error deleting data from Cosmos DB: {str(e)}")
            return False
    
    def get_data(self, company_id: str = None, report_type: str = None) -> list:
        try:
            if company_id and report_type:
                return self.container.query_items(
                    query=f"SELECT * FROM c WHERE c.companyId = '{company_id}' AND c.reportType = '{report_type}'",
                    enable_cross_partition_query=True
                )
            elif company_id:
                return self.container.query_items(
                    query=f"SELECT * FROM c WHERE c.companyId = '{company_id}'",
                    enable_cross_partition_query=True
                )   
            elif report_type:
                return self.container.query_items(
                    query=f"SELECT * FROM c WHERE c.reportType = '{report_type}'",
                    enable_cross_partition_query=True
                )
            else:
                return self.container.query_items(
                    query=f"SELECT * FROM c",
                    enable_cross_partition_query=True
                )
        except Exception as e:
            logger.error(f"Error getting data from Cosmos DB: {str(e)}")
            return []

if __name__ == "__main__":
    try:
        loader = CosmosDBLoader(container_name="schedules")
        loader.create_container()   
        # Use the path to your generated JSON file
        json_file_path = os.path.join(os.path.dirname(__file__), "data/companyID_schedules.json")
        loader.upload_data(json_file_path)
    except ValueError as e:
        logger.error(f"Configuration error: {str(e)}")
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
    
    # loader = CosmosDBLoader(container_name="schedules")
    # loader.delete_data("AMZN")
