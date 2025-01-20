import logging
import os
import requests 
from datetime import datetime, timezone 
from typing import List, Dict, Optional
from tenacity import retry, stop_after_attempt, wait_exponential
import azure.functions as func  
from azure.cosmos import CosmosClient
from azure.identity import DefaultAzureCredential
# logger setting 
logging.basicConfig(level=logging.INFO) 
logger = logging.getLogger(__name__)


""" 
Monthly Scheduler: 
1. The trigger will run at 8am EST every 1st date of the month 
2. It will loop through all the reports in the MONTHLY_REPORTS list 
3. Then it will create a report for each monthly report type using the /api/reports/generate/curation endpoint 
4. Once done, the report will be uploaded to blob storage 

To do:
5. trigger the email function to an email to send to admin the link to the report that has been created 
6. automatically send the report to user using the blob link got from the curation endpoint 
check other way to use endpoint 

"""

MONTHLY_REPORTS = ['Monthly_Economics', 'Ecommerce', "Home_Improvement", "Company_Analysis"]

COMPANY_NAME = ["Home Depot", "Lowes"]

CURATION_REPORT_ENDPOINT = f'{os.environ["WEB_APP_URL"]}/api/reports/generate/curation'

EMAIL_ENDPOINT = f'{os.environ["WEB_APP_URL"]}/api/reports/digest'

TIMEOUT_SECONDS = 300

MAX_RETRIES = 3

class CosmoDBManager:
    def __init__(self, container_name: str, 
                 db_uri: str, 
                 credential: str, 
                 database_name: str):
        self.container_name = container_name
        self.db_uri = db_uri
        self.credential = credential
        self.database_name = database_name

        if not all([self.container_name, self.db_uri, self.credential, self.database_name]):
            raise ValueError("Missing required environment variables for Cosmos DB connection")

        self.client = CosmosClient(url=self.db_uri, credential=self.credential, consistency_level="Session")
        self.database = self.client.get_database_client(self.database_name)
        self.container = self.database.get_container_client(self.container_name)
    
    def get_email_list(self) -> List[str]:
        query = "SELECT * FROM c where c.isActive = true"
        items = self.container.query_items(query, enable_cross_partition_query=True)
        email_list: List[str] = []
        for item in items:
            if "email" in item:
                email_list.append(item['email'])
        return email_list

def generate_report(report_topic: str, company_name: Optional[str] = None) -> Optional[Dict]:
    """Generate a report and return the response if successful """

    payload = {
        'report_topic': report_topic
    }

    if company_name:
        payload['company_name'] = company_name

    @retry(stop = stop_after_attempt(MAX_RETRIES), wait = wait_exponential(multiplier=1, min=4, max=10))
    def _make_report_request():
        logger.debug(f"Sending request to generate report for {report_topic}")
        response = requests.post(
            CURATION_REPORT_ENDPOINT,
            headers={"Content-Type": "application/json"},
            json=payload, 
            timeout=TIMEOUT_SECONDS
        )
        logger.debug(f"Received response for report generation request for {report_topic}")
        return response.json()

    try:
        report_response = _make_report_request()
        logger.info(f"Report generation response for {report_topic}: {report_response}")
        return report_response
    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to generate report for {report_topic}: {str(e)}")
        return None

def send_report_email(blob_link: str, report_name: str) -> bool:
    """Send email with report link and return success status """

    container_name = 'subscription_emails'
    db_uri = f"https://{os.environ['AZURE_DB_ID']}.documents.azure.com:443/" if os.environ.get('AZURE_DB_ID') else None
    credential = DefaultAzureCredential()
    database_name = os.environ.get('AZURE_DB_NAME') if os.environ.get('AZURE_DB_NAME') else None

    cosmo_db_manager = CosmoDBManager(
        container_name=container_name,
        db_uri=db_uri,
        credential=credential,
        database_name=database_name
    )
    email_list = cosmo_db_manager.get_email_list()

    email_payload = {
        'blob_link': blob_link,
        'email_subject': 'Sales Factory Monthly Report',
        'recipients': email_list,
        'save_email': 'yes'
    }

    try: 
        logger.debug(f"Sending email for report {report_name} with blob link {blob_link}")
        response = requests.post(
            EMAIL_ENDPOINT,
            headers={"Content-Type": "application/json"},
            json=email_payload,
            timeout=TIMEOUT_SECONDS
        )
        response_json = response.json()
        logger.info(f"Email response for {report_name}: {response_json}")

        if response_json.get('status') == 'error':
            raise requests.exceptions.RequestException(response_json.get('message'))
        
        return response_json.get('status') == 'success'
    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to send email for {report_name}: {str(e)}")
        return False

def main(mytimer: func.TimerRequest) -> None: 

    utc_timestamp = datetime.now(timezone.utc).isoformat()
    logger.info(f"Monthly report generation started at {utc_timestamp}")

    for report in MONTHLY_REPORTS:
        if report == "Company_Analysis":
            for company in COMPANY_NAME:
                logger.info(f"Generating company report for {company} at {utc_timestamp}")
                response_json = generate_report(report, company)
        else:
            logger.info(f"Generating report for {report} at {utc_timestamp}")
            response_json = generate_report(report)

        if not response_json or response_json.get('status') != 'success':
            logger.error(f"Failed to generate report for {report} at {utc_timestamp}")
            continue

        # extract blob link and send email 
        blob_link = response_json.get('report_url')

        if not blob_link:
            logger.error(f"Failed to extract blob link for {report} at {utc_timestamp}")
            continue 

        if send_report_email(blob_link, report):
            logger.info(f"Report {report} sent successfully at {utc_timestamp}")
        else:
            logger.error(f"Failed to send email for {report} at {utc_timestamp}")
            logger.error(f"Report with blob link {blob_link} was not sent")

    logger.info(f"Monthly report generation completed at {datetime.now(timezone.utc).isoformat()}")


