import datetime
import logging
import json
import os
from azure.cosmos import CosmosClient
from datetime import datetime, timedelta
import azure.functions as func
from shared.cosmos_db import get_active_schedules, update_last_run


def should_trigger_fetch(schedule):
    """Determine if fetch should be triggered based on schedule"""
    last_run = datetime.fromisoformat(schedule.get('lastRun', '2025-01-01'))
    frequency = schedule['frequency']  # 'weekly' or 'monthly'
    
    now = datetime.utcnow()
    if frequency == 'weekly':
        next_run = last_run + timedelta(days=7)
    else:  # monthly
        # Add one month (approximately)
        # TODO: Improve this to be more accurate
        next_run = last_run + timedelta(days=30)
        
    return now >= next_run

def trigger_document_fetch(schedule):
    """Queue the document fetch task based on schedule configuration"""
    
    # Create message
    message = {
        'scheduleId': schedule['id'],
        'reportType': schedule['reportType'],
        'companyId': schedule['companyId'],
        'timestamp': datetime.utcnow().isoformat()
    }
    
    # TODO: Implement document fetch logic
    # This will be implemented when document fetch logic is ready
    logging.info(f"Queued fetch task: {json.dumps(message)}")
    
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
