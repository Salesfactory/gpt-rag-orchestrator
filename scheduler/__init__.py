import datetime
import logging
import json
import os
from azure.cosmos import CosmosClient
from datetime import datetime, timedelta,UTC
import azure.functions as func
from shared.cosmos_db import get_active_schedules, update_last_run
from shared.cosmo_data_loader import CosmosDBLoader
import requests


def should_trigger_fetch(schedule):
    """Determine if fetch should be triggered based on schedule"""
    # Convert last_run to timezone-aware datetime
    last_run = datetime.fromisoformat(schedule.get('lastRun', '2025-01-01')).replace(tzinfo=UTC)
    frequency = schedule['frequency']
    
    now = datetime.now(UTC)
    if frequency == 'weekly':
        next_run = last_run + timedelta(days=7)
    elif frequency == 'monthly':
        next_run = last_run + timedelta(days=30)
    elif frequency == 'twice_daily':
        # Check if it's been at least 12 hours since last run
        next_run = last_run + timedelta(hours=12)
        # Only trigger at 12 AM or 12 PM UTC
        if now.hour not in [0, 12]:
            return False
    elif frequency == 'every_minute':
        next_run = last_run + timedelta(minutes=1)
        
    return now >= next_run

def trigger_document_fetch(schedule):
    """Queue the document fetch task based on schedule configuration"""
    
    # Create message
    message = {
        'scheduleId': schedule['id'],
        'reportType': schedule['reportType'],
        'companyId': schedule['companyId'],
        'timestamp': datetime.now(UTC).isoformat()
    }
    
    # Create payload for the API endpoint
    # after_date = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
    after_date = (datetime.now(UTC)).strftime('%Y-%m-%d')
    payload = {
        "equity_id": schedule['companyId'],
        "filing_type": schedule['reportType'],
        "after_date": after_date
    }
    
    # Make API request
    try:
        response = requests.post(
            "https://webgpt0-vm2b2htvuuclm.azurewebsites.net/api/SECEdgar/financialdocuments/process-and-summarize",
            json=payload
        )
        response.raise_for_status()
        logging.info(f"Successfully triggered document fetch: {json.dumps(message)}")
    except requests.exceptions.RequestException as e:
        logging.error(f"Failed to trigger document fetch: {str(e)}")

    # TODO: if failed then skip the update_last_run 
    
    # Update last run timestamp
    update_last_run(schedule['id'])



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
        active_schedules = cosmos_data_loader.get_data()
        for schedule in active_schedules:
            if should_trigger_fetch(schedule):
                logging.info(f"Triggering fetch for schedule {schedule['id']}")
                trigger_document_fetch(schedule)
            else:
                logging.info(f"Schedule {schedule['id']} not due for execution")
                
    except Exception as e:
        logging.error(f"Error in scheduler: {str(e)}")
