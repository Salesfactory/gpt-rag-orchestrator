import logging
import os
import requests 
from datetime import datetime, timezone 
from typing import List, Dict, Optional
from tenacity import retry, stop_after_attempt, wait_exponential
import azure.functions as func
from azure.identity import DefaultAzureCredential
from shared.cosmo_data_loader import CosmosDBLoader
# logger setting 
logging.basicConfig(level=logging.INFO) 
logger = logging.getLogger(__name__)

""" 
Task: 
1. This is a weekly scheduler that will run every week on monday at 13:00 PM UTC -> 8:00 AM EST
2. It will loop through all the reports in the WEEKLY_REPORTS list 
3. Then it will create a report for each weekly report type using the /api/reports/generate/curation endpoint 
4. Once done, the report will be uploaded to blob storage 

To do:
5. trigger the email function to an email to send to admin the link to the report that has been created 
"""


WEEKLY_REPORTS = ['Weekly_Economics'] 

CURATION_REPORT_ENDPOINT = f'{os.environ["WEB_APP_URL"]}/api/reports/generate/curation'

EMAIL_ENDPOINT = f'{os.environ["WEB_APP_URL"]}/api/reports/digest'

TIMEOUT_SECONDS = 300

MAX_RETRIES = 3

def generate_report(report_topic: str, company_name: Optional[str]) -> Optional[Dict]:
    """Generate a report and return the response if successful """

    payload = {
        'report_topic': report_topic,
        'company_name': company_name
    }

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

def send_report_email(blob_link: str, report_name: str, organization_name: str, email_list: List[str]) -> bool:
    """Send email with report link and return success status """

    email_subject = f"{organization_name} Weekly Report"

    email_payload = {
        'blob_link': blob_link,
        'email_subject': email_subject,
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
    logger.info(f"Weekly report generation started at {utc_timestamp}")

    container_name = 'subscription_emails'
    db_uri = f"https://{os.environ['AZURE_DB_ID']}.documents.azure.com:443/" if os.environ.get('AZURE_DB_ID') else None
    credential = DefaultAzureCredential()
    database_name = os.environ.get('AZURE_DB_NAME') if os.environ.get('AZURE_DB_NAME') else None

    logger.info(f"Initializing Cosmos DB Loader")

    cosmo_db_manager = CosmosDBLoader(
        container_name=container_name,
        db_uri=db_uri,
        credential=credential,
        database_name=database_name
    )

    logger.info(f"Getting organization email list from Cosmos DB")

    organizations = cosmo_db_manager.get_email_list_by_org()

    for report in WEEKLY_REPORTS:
        for organization_name, email_list in organizations.items():
            logger.info(f"Generating {report} for organization: {organization_name} at {utc_timestamp}")
            response_json = generate_report(report, organization_name)

            if not response_json or response_json.get('status') != 'success':
                logger.error(f"Failed to generate report for {report} at {utc_timestamp}")
                continue

            # extract blob link and send email 
            blob_link = response_json.get('report_url')

            if not blob_link:
                logger.error(f"Failed to extract blob link for {report} at {utc_timestamp}")
                continue 

            logger.info(f"Sending email for {report} at {utc_timestamp}")

            if send_report_email(blob_link, report, organization_name, email_list):
                logger.info(f"Report {report} sent successfully at {utc_timestamp}")
            else:
                logger.error(f"Failed to send email for {report} at {utc_timestamp}")
                logger.error(f"Report with blob link {blob_link} was not sent")

    logger.info(f"Weekly report generation completed at {datetime.now(timezone.utc).isoformat()}")

