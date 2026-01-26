import os
import logging
from datetime import datetime, UTC, timedelta, timezone
from typing import List, Dict, Any
from azure.cosmos import CosmosClient, exceptions
from azure.core import MatchConditions
from azure.identity import DefaultAzureCredential
from dotenv import load_dotenv

load_dotenv()


def _client():
    db_id = os.getenv("AZURE_DB_ID")
    if not db_id:
        raise ValueError("AZURE_DB_ID environment variable not set")
    return CosmosClient(
        url=f"https://{db_id}.documents.azure.com:443/",
        credential=DefaultAzureCredential(),
    )


def cosmos_container():
    db = os.getenv("AZURE_DB_NAME")
    if not db:
        raise ValueError("AZURE_DB_NAME environment variable not set")
    cont = os.getenv("COSMOS_CONTAINER", "reportJobs")
    return _client().get_database_client(db).get_container_client(cont)


async def load_scheduled_jobs() -> List[Dict[str, Any]]:
    """
    Returns jobs that are due now.
    MUST include _etag -> mapped to 'etag' for the activity.
    """
    c = cosmos_container()
    query = "SELECT c.id, c.job_id, c.organization_id, c.tenant_id, c.schedule_time, c.status, c._etag FROM c WHERE c.status = 'QUEUED' AND c.schedule_time <= @now"
    params = [{"name": "@now", "value": datetime.now(UTC).isoformat()}]
    items = list(
        c.query_items(query=query, parameters=params, enable_cross_partition_query=True)
    )

    # Normalize fields the orchestrators/activities expect
    jobs = []
    for it in items:
        jobs.append(
            {
                "job_id": it.get("job_id") or it.get("id"),
                "organization_id": it["organization_id"],
                "tenant_id": it["tenant_id"],
                "etag": it.get("_etag"),
            }
        )
    return jobs


def try_mark_job_running(
    container,
    job_id: str,
    organization_id: str,
    etag: str,
    processing_instance_id: str | None = None,
) -> bool:
    """
    Optimistic transition QUEUED -> RUNNING using ETag to avoid duplicates.
    Optionally records a processing_instance_id for retry ownership.

    Returns:
        True if successfully marked as running
        False if job already taken (etag mismatch) or not found (404)
    """
    try:
        # Load existing doc to preserve its body (replace only status)
        existing = container.read_item(item=job_id, partition_key=organization_id)
        now_iso = datetime.now(UTC).isoformat()
        existing["status"] = "RUNNING"
        existing["updated_at"] = now_iso
        if not existing.get("started_at"):
            existing["started_at"] = now_iso
        if processing_instance_id:
            existing["processing_instance_id"] = processing_instance_id
        container.replace_item(
            item=job_id,
            body=existing,
            etag=etag,
            match_condition=MatchConditions.IfNotModified,
        )
        return True
    except exceptions.CosmosHttpResponseError as e:
        if e.status_code == 412:
            logging.info(f"[Idempotency] Job {job_id} already taken (etag mismatch).")
            return False
        elif e.status_code == 404:
            logging.warning(
                f"[Cosmos] Job {job_id} not found for organization {organization_id} - likely already processed or deleted"
            )
            return False
        raise


def mark_job_result(
    container, job_id: str, organization_id: str, status: str, error: str = None
):
    try:
        existing = container.read_item(item=job_id, partition_key=organization_id)
        existing["status"] = status
        existing["completed_at"] = datetime.now(UTC).isoformat()
        if error:
            existing["error"] = error
        container.replace_item(item=job_id, body=existing)
    except exceptions.CosmosHttpResponseError as e:
        if e.status_code == 404:
            logging.warning(
                f"[Cosmos] Job {job_id} not found for organization {organization_id}, skipping mark_job_result"
            )
        else:
            raise
    except Exception as e:
        logging.error(f"[Cosmos] mark_job_result failed for {job_id}: {e}")


def acquire_global_lease(
    container,
    instance_id: str,
    ttl_minutes: int,
) -> bool:
    """
    Acquire a single global lease document to enforce one-job-at-a-time processing.
    Returns True if acquired, False otherwise.
    """
    lease_id = "global_lease"
    partition_key = "global"
    now = datetime.now(timezone.utc)
    expires_at = (now + timedelta(minutes=ttl_minutes)).isoformat()

    try:
        lease = container.read_item(item=lease_id, partition_key=partition_key)
        lease_expiry = datetime.fromisoformat(lease.get("expires_at", "2000-01-01"))

        if now <= lease_expiry:
            return False

        lease["owner_instance_id"] = instance_id
        lease["expires_at"] = expires_at

        container.replace_item(
            item=lease_id,
            body=lease,
            etag=lease.get("_etag"),
            match_condition=MatchConditions.IfNotModified,
        )
        return True
    except exceptions.CosmosResourceNotFoundError:
        lease_doc = {
            "id": lease_id,
            "organization_id": partition_key,
            "owner_instance_id": instance_id,
            "expires_at": expires_at,
        }
        try:
            container.create_item(body=lease_doc)
            return True
        except exceptions.CosmosResourceExistsError:
            return False
    except exceptions.CosmosHttpResponseError as e:
        if e.status_code == 412:
            return False
        raise


def release_global_lease(container, instance_id: str) -> None:
    lease_id = "global_lease"
    partition_key = "global"
    try:
        lease = container.read_item(item=lease_id, partition_key=partition_key)
        if lease.get("owner_instance_id") == instance_id:
            container.delete_item(item=lease_id, partition_key=partition_key)
    except exceptions.CosmosResourceNotFoundError:
        pass


def is_global_lease_active(container) -> bool:
    lease_id = "global_lease"
    partition_key = "global"
    try:
        lease = container.read_item(item=lease_id, partition_key=partition_key)
        lease_expiry = datetime.fromisoformat(lease.get("expires_at", "2000-01-01"))
        return datetime.now(timezone.utc) <= lease_expiry
    except exceptions.CosmosResourceNotFoundError:
        return False


def reset_stale_running_jobs(container, cutoff_iso: str) -> List[Dict[str, Any]]:
    """
    Reset RUNNING jobs older than cutoff back to QUEUED.
    Returns a list of job payloads for requeue.
    """
    query = (
        "SELECT c.id, c.job_id, c.organization_id, c.updated_at, c.started_at "
        "FROM c WHERE c.status = 'RUNNING' AND c.updated_at <= @cutoff"
    )
    params = [{"name": "@cutoff", "value": cutoff_iso}]
    items = list(
        container.query_items(
            query=query, parameters=params, enable_cross_partition_query=True
        )
    )

    reset_jobs: List[Dict[str, Any]] = []
    now_iso = datetime.now(timezone.utc).isoformat()
    for item in items:
        try:
            job_id = item.get("id") or item.get("job_id")
            organization_id = item.get("organization_id")
            if not job_id or not organization_id:
                continue

            job_doc = container.read_item(item=job_id, partition_key=organization_id)
            if job_doc.get("status") != "RUNNING":
                continue

            job_doc["status"] = "QUEUED"
            job_doc["updated_at"] = now_iso
            job_doc["stale_reset_at"] = now_iso

            updated = container.replace_item(item=job_id, body=job_doc)
            reset_jobs.append(
                {
                    "job_id": updated.get("job_id") or updated.get("id"),
                    "organization_id": updated.get("organization_id"),
                    "tenant_id": updated.get("tenant_id"),
                    "etag": updated.get("_etag"),
                }
            )
        except Exception as e:
            logging.error(f"[Cosmos] Failed to reset stale job: {e}")

    return reset_jobs
