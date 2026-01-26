import json
import os
from typing import Any, Dict, Optional

from azure.storage.queue import QueueClient


def _get_queue_client(queue_name: str) -> QueueClient:
    connection_string = os.getenv("AzureWebJobsStorage")
    if not connection_string:
        raise ValueError("AzureWebJobsStorage is not set")
    api_version = None
    if (
        "UseDevelopmentStorage=true" in connection_string
    ):
        # Use an older, widely supported version for Azurite
        api_version = "2020-10-02"
    return QueueClient.from_connection_string(
        conn_str=connection_string,
        queue_name=queue_name,
        api_version=api_version,
    )


def enqueue_message(
    queue_name: str,
    payload: Dict[str, Any],
    visibility_timeout: Optional[int] = None,
) -> None:
    client = _get_queue_client(queue_name)
    message = json.dumps(payload)
    if visibility_timeout is None:
        client.send_message(message)
    else:
        client.send_message(message, visibility_timeout=visibility_timeout)
