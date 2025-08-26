import logging
from datetime import datetime, timezone
from shared.util import WEB_APP_URL
from azure.functions import func

logger = logging.getLogger(__name__)

def main(mytimer: func.TimerRequest) -> None:
    """Main function for report scheduler - runs at 2:00 AM UTC daily"""
    
    # Check if the environment variable is set
    if not WEB_APP_URL:
        logger.error("WEB_APP_URL environment variable not set")
        return
    
    start_time = datetime.now(timezone.utc)
    logger.info(f"Report scheduler started at {start_time}")
    
    try:
        print("Report scheduler started")
    finally:
        end_time = datetime.now(timezone.utc)
        duration = end_time - start_time
        logger.info(
            f"Report scheduler completed at {end_time}. Duration: {duration}"
        )
