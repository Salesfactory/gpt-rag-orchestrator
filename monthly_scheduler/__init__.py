import logging
import os
import requests 
from datetime import datetime, timezone 
import azure.functions as func

MONTHLY_REPORTS = ['Monthly_Economics', "Ecommerce"] 

CURATION_REPORT_ENDPOINT = 'https://webgpt0-vm2b2htvuuclm.azurewebsites.net/api/reports/generate/curation'

def main(mytimer: func.TimerRequest) -> None:
    utc_timestamp = datetime.now(timezone.utc).isoformat()

    for report in MONTHLY_REPORTS:
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
                # todo: extract the report email from blob storage 

            else:
                logging.error(f"Failed to generate report for {report}: {response_json.get('message')} at {utc_timestamp}")
        except requests.exceptions.RequestException as e:
            logging.error(f'Failed to generate report for {report}: {str(e)} at {utc_timestamp}')


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