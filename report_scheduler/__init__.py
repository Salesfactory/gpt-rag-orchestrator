import logging
import os
import requests
from datetime import datetime, timezone
from azure.functions import func
from tenacity import retry, wait_exponential, stop_after_attempt

logger = logging.getLogger(__name__)

# Environment variables
WEB_APP_URL = os.getenv("WEB_APP_URL", None)
TIMEOUT_SECONDS = 120
MAX_RETRIES = 3

def main(mytimer: func.TimerRequest) -> None:
    """Main function for report scheduler - runs at 2:00 AM UTC daily"""
    
    # Check if the environment variable is set
    if not WEB_APP_URL:
        logger.error("WEB_APP_URL environment variable not set")
        return
    
    start_time = datetime.now(timezone.utc)
    logger.info(f"Report scheduler started at {start_time}")
    
    try:
        logger.info("Starting HTTP requests to endpoints")
        
        
        full_url = f"{WEB_APP_URL}/api/report-jobs"
        logger.info(f"Sending request to: {full_url}")
        
        try:
            response = send_http_request(full_url)
            log_response_result(full_url, response)
        except Exception as e:
            logger.error(f"Failed to send request to {full_url}: {str(e)}")
                
    except Exception as e:
        logger.error(f"Unexpected error in report scheduler: {str(e)}")
    finally:
        end_time = datetime.now(timezone.utc)
        duration = end_time - start_time
        logger.info(
            f"Report scheduler completed at {end_time}. Duration: {duration}"
        )

@retry(
    stop=stop_after_attempt(MAX_RETRIES),
    wait=wait_exponential(multiplier=1, min=4, max=10),
)
def send_http_request(url: str) -> requests.Response:
    """Send HTTP request with retry logic"""
    logger.debug(f"Sending HTTP request to {url}")
    
    # Send a simple GET request to check if the endpoint is available
    response = requests.get(
        url,
        headers={"Content-Type": "application/json"},
        timeout=TIMEOUT_SECONDS,
    )
    
    logger.debug(f"Received response from {url}: {response.status_code}")
    return response

def log_response_result(endpoint: str, response: requests.Response) -> None:
    """Log the results of the endpoint call"""
    try:
        response_json = response.json() if response.content else {}
    except Exception:
        response_json = {"error": "Could not parse JSON response"}
    
    log_data = {
        "endpoint": endpoint,
        "status_code": response.status_code,
        "response_time": response.elapsed.total_seconds(),
        "response_size": len(response.content),
        "response_data": response_json
    }
    
    if response.status_code == 200:
        logger.info(f"Successfully called {endpoint}: {log_data}")
    elif response.status_code in [401, 403]:
        logger.warning(f"Authentication/Authorization issue with {endpoint}: {log_data}")
    elif response.status_code in [404, 405]:
        logger.warning(f"Endpoint not found or method not allowed for {endpoint}: {log_data}")
    elif response.status_code >= 500:
        logger.error(f"Server error for {endpoint}: {log_data}")
    else:
        logger.warning(f"Unexpected status code for {endpoint}: {log_data}")
