import os
import logging
from datetime import datetime
from typing import List, Dict, Any
from azure.cosmos import CosmosClient, exceptions


def _client():
    return CosmosClient(url=os.getenv("COSMOS_URL"), credential=os.getenv("COSMOS_KEY"))


def cosmos_container():
    db = os.getenv("COSMOS_DB", "reports")
    cont = os.getenv("COSMOS_CONTAINER", "jobs")
    return _client().get_database_client(db).get_container_client(cont)


async def load_scheduled_jobs() -> List[Dict[str, Any]]:
    """
    Returns jobs that are due now.
    MUST include _etag -> mapped to 'etag' for the activity.
    """
    c = cosmos_container()
    query = "SELECT c.id, c.job_id, c.organization_id, c.tenant_id, c.schedule_time, c.status, c._etag FROM c WHERE c.status = 'QUEUED' AND c.schedule_time <= @now"
    params = [{"name": "@now", "value": datetime.utcnow().isoformat()}]
    items = list(c.query_items(query=query, parameters=params, enable_cross_partition_query=True))

    # Normalize fields the orchestrators/activities expect
    jobs = []
    for it in items:
        jobs.append({
            "job_id": it.get("job_id") or it.get("id"),
            "organization_id": it["organization_id"],
            "tenant_id": it["tenant_id"],
            "etag": it.get("_etag")
        })
    return jobs


def try_mark_job_running(container, job_id: str, etag: str) -> bool:
    """
    Optimistic transition QUEUED -> RUNNING using ETag to avoid duplicates.
    """
    try:
        # Load existing doc to preserve its body (replace only status)
        existing = container.read_item(item=job_id, partition_key=job_id)
        existing["status"] = "RUNNING"
        container.replace_item(item=job_id, body=existing, etag=etag, match_condition="IfMatch")
        return True
    except exceptions.CosmosHttpResponseError as e:
        if e.status_code == 412:
            logging.info(f"[Idempotency] Job {job_id} already taken (etag mismatch).")
            return False
        raise


def mark_job_result(container, job_id: str, status: str, error: str = None):
    try:
        existing = container.read_item(item=job_id, partition_key=job_id)
        existing["status"] = status
        existing["completed_at"] = datetime.utcnow().isoformat()
        if error:
            existing["error"] = error
        container.replace_item(item=job_id, body=existing)
    except Exception as e:
        logging.error(f"[Cosmos] mark_job_result failed for {job_id}: {e}")
