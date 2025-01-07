import logging
import os
import requests 
from datetime import datetime, timezone 
import azure.functions as func

WEEKLY_REPORTS = ['Weekly_Economics'] # for now only weekly economics is supported 

CURATION_REPORT_ENDPOINT = 'https://webgpt0-vm2b2htvuuclm.azurewebsites.net/api/reports/generate/curation'

def main(mytimer: func.TimerRequest) -> None:
    utc_timestamp = datetime.now(timezone.utc).isoformat()

    for report in WEEKLY_REPORTS:
        try: 
            # call the report generation API 
            payload = {
                'report_topic': report
            }
            response = requests.post(
                CURATION_REPORT_ENDPOINT,
                headers={"Content-Type": "application/json"},
                json=payload, 
                timeout=300
            )
            response_json = response.json()
            if response_json.get('status') == 'success':
                logging.info(f"Report {report} generated successfully at {utc_timestamp}")
                # todo: extract the report email from blob storage and 

            else:
                logging.error(f"Failed to generate report for {report}: {response_json.get('message')} at {utc_timestamp}")
        except requests.exceptions.RequestException as e:
            logging.error(f'Failed to generate report for {report}: {str(e)} at {utc_timestamp}')

""" 
Task: 
1. This is a weekly scheduler that will run every week on monday at 13:00 PM UTC -> 8:00 AM EST
2. It will loop through all the reports in the WEEKLY_REPORTS list 
3. Then it will create a report for each weekly report type using the /api/reports/generate/curation endpoint 
4. Once done, the report will be uploaded to blob storage 

To do:
5. trigger the email function to an email to send to admin the link to the report that has been created 
"""
