import datetime
import logging
import json
import os
from azure.cosmos import CosmosClient
from datetime import datetime, timedelta, UTC
import azure.functions as func
from shared.cosmos_db import get_active_schedules, update_last_run
import requests


def should_trigger_fetch(schedule):
    """Determine if fetch should be triggered based on schedule"""
    last_run = datetime.fromisoformat(schedule.get('lastRun', '2025-01-01'))
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
    after_date = datetime.now(UTC) 
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
    try:
        active_schedules = get_active_schedules()
        for schedule in active_schedules:
            if should_trigger_fetch(schedule):
                logging.info(f"Triggering fetch for schedule {schedule['id']}")
                trigger_document_fetch(schedule)
            else:
                logging.info(f"Schedule {schedule['id']} not due for execution")
                
    except Exception as e:
        logging.error(f"Error in scheduler: {str(e)}")
