import datetime
import logging
import json
import os
from azure.cosmos import CosmosClient
from datetime import datetime, timedelta,UTC
import azure.functions as func
from shared.cosmos_db import get_active_schedules, was_summarized_today
from shared.cosmo_data_loader import CosmosDBLoader
import requests

def trigger_document_fetch(schedule: dict) -> None:
    """ 
    Queue the document fetch task based on schedule configuration 

    Args: 
       schedule (dict): schedule document containing companyId, reportType, frequency, lastRun, and some other attributes
    """
    cosmos_data_loader = CosmosDBLoader('schedules')
    process_and_summarize_url = os.getenv('PROCESS_AND_SUMMARIZE_URL')
    
    # create payload for the API endpoint 
    start_time = datetime.now(UTC).isoformat()
    
    payload = {
        "equity_id": schedule['companyId'],
        "filing_type": schedule['reportType'],
        "after_date": '2024-12-29'
    }
    try: 
        if was_summarized_today(schedule):
            logging.info(f"Skipping document fetch for {schedule['companyId']} {schedule['reportType']} as it was already summarized today")
            schedule['summarized_today'] = False # reset the summarized_today flag
        else: 
            response = requests.post(
                process_and_summarize_url,
                headers={
                    "Content-Type": "application/json"
                },
                json=payload,
                timeout = 300
            )

            response_json = response.json()
            schedule['summarized_today'] = (response_json.get('code', 500) == 200)

            # log the code outcome
            logging.info(f"Code: {response_json.get('code')}")
            if schedule['summarized_today']:
                logging.info(f"Successfully triggered document fetch for: {json.dumps(schedule['companyId'])} - {json.dumps(schedule['reportType'])}")
            else:
                if response_json.get('code') == 404:
                    logging.error(f"No new uploaded documents found for: {json.dumps(schedule['companyId'])} - {json.dumps(schedule['reportType'])}. Last checked time: {start_time}")
                else:
                    logging.error(f"Failed to trigger document fetch for: {json.dumps(schedule['companyId'])} - {json.dumps(schedule['reportType'])}")
    
        # update last run time regarless of the outcome 
        schedule['lastRun'] = start_time
        cosmos_data_loader.update_last_run(schedule)
    except Exception as e:
        logging.error(f"Error in trigger_document_fetch: {str(e)}")
        schedule['summarized_today'] = False


def main(timer: func.TimerRequest) -> None:

    # schedule cosmos attributes 
    # id
    # lastRun
    # frequency
    # companyId
    # reportType
    # isActive

    # Main scheduling logic
    cosmos_data_loader = CosmosDBLoader('schedules')
    try:
        # get all schedules that are active and have a frequency of twice_a_day
        active_schedules = cosmos_data_loader.get_data(frequency="twice_a_day")
        for schedule in active_schedules:
            logging.info(f"Triggering fetch for schedule {schedule['id']}")
            trigger_document_fetch(schedule)
                
    except Exception as e:
        logging.error(f"Error in scheduler: {str(e)}")
