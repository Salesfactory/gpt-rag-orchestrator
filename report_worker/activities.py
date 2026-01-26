"""
Durable Functions Activities for report generation.

Activities are the actual work units that get executed by the orchestrator.
Each activity should be idempotent and deterministic.
"""

import logging
import traceback
import asyncio
import os
from datetime import datetime, timezone
from azure.cosmos import exceptions

from durable_functions_registry import app
from report_worker.processor import (
    process_report_job,
    ReportJobDeterministicError,
    ReportJobTransientError,
)
from shared.util import update_report_job_status

from shared.cosmos_jobs import cosmos_container, try_mark_job_running, mark_job_result

logger = logging.getLogger(__name__)


def _get_job_timeout_seconds() -> int:
    raw_value = os.getenv("REPORT_JOB_TIMEOUT_SECONDS", "600")
    try:
        timeout = int(raw_value)
        return max(timeout, 1)
    except ValueError:
        logger.warning(
            f"[GenerateReportActivity] Invalid REPORT_JOB_TIMEOUT_SECONDS={raw_value}, defaulting to 600"
        )
        return 600


JOB_TIMEOUT_SECONDS = _get_job_timeout_seconds()


def _get_error_type(job_doc: dict) -> str | None:
    error = job_doc.get("error")
    if isinstance(error, dict):
        return error.get("error_type")
    return None


def _owner_matches(
    existing_owner: str | None, processing_instance_id: str | None
) -> bool:
    if not processing_instance_id:
        return False
    return existing_owner == processing_instance_id


@app.activity_trigger(input_name="job")
async def GenerateReportActivity(job: dict) -> dict:
    """
    Durable Activity: Process a single report generation job.

    This activity wraps the existing process_report_job logic with:
    - Timeout protection (configurable)
    - Comprehensive error handling
    - Status tracking
    - Idempotency support

    Args:
        job: Dictionary containing:
            - job_id: Unique job identifier
            - organization_id: Organization/tenant ID
            - tenant_id: Tenant identifier (optional)
            - etag: Cosmos DB _etag for idempotency (optional)
            - attempt: Attempt number (for retries)
            - processing_instance_id: Durable instance identifier (for retry ownership)

    Returns:
        Dictionary with job result:
            - job_id: Job identifier
            - organization_id: Organization ID
            - status: "SUCCEEDED", "FAILED", or "SKIPPED"
            - completed_at: ISO timestamp (if succeeded)
            - error: Error message (if failed)
            - reason: Skip reason (if skipped)
    """
    job_id = job["job_id"]
    organization_id = job["organization_id"]
    etag = job.get("etag")
    attempt = job.get("attempt", 1)
    processing_instance_id = job.get("processing_instance_id")

    logger.info(
        f"[GenerateReportActivity] Starting job {job_id} for org {organization_id} (attempt {attempt})"
    )

    try:
        if etag:
            container = cosmos_container()
            claimed = try_mark_job_running(
                container,
                job_id,
                organization_id,
                etag,
                processing_instance_id=processing_instance_id,
            )
            if not claimed:
                # Job either taken by another worker or doesn't exist
                try:
                    existing = container.read_item(
                        item=job_id, partition_key=organization_id
                    )
                    existing_status = (existing.get("status") or "").upper()
                    existing_owner = existing.get("processing_instance_id")
                    existing_error_type = _get_error_type(existing)

                    if existing_status == "RUNNING" and _owner_matches(
                        existing_owner, processing_instance_id
                    ):
                        logger.info(
                            f"[GenerateReportActivity] Job {job_id} already RUNNING for this instance, continuing."
                        )
                    elif (
                        existing_status == "FAILED"
                        and existing_error_type
                        in {
                            "transient",
                            "timeout",
                        }
                        and _owner_matches(existing_owner, processing_instance_id)
                    ):
                        logger.info(
                            f"[GenerateReportActivity] Retrying transient failure for job {job_id}."
                        )
                    else:
                        logger.info(
                            f"[GenerateReportActivity] Skip {job_id}, another worker took it."
                        )
                        return {
                            "job_id": job_id,
                            "organization_id": organization_id,
                            "status": "SKIPPED",
                            "reason": "Job already taken by another worker",
                        }
                except exceptions.CosmosHttpResponseError as e:
                    if e.status_code == 404:
                        logger.warning(
                            f"[GenerateReportActivity] Job {job_id} not found in database - skipping"
                        )
                        return {
                            "job_id": job_id,
                            "organization_id": organization_id,
                            "status": "SKIPPED",
                            "reason": "Job not found in database",
                        }
                    raise

        # Add timeout protection (configurable per job)
        async with asyncio.timeout(JOB_TIMEOUT_SECONDS):
            await process_report_job(job_id, organization_id, attempt, allow_retry=True)

        completion_time = datetime.now(timezone.utc).isoformat()
        logger.info(f"[GenerateReportActivity] Successfully completed job {job_id}")

        # mark success
        mark_job_result(cosmos_container(), job_id, organization_id, status="SUCCEEDED")

        return {
            "job_id": job_id,
            "organization_id": organization_id,
            "status": "SUCCEEDED",
            "completed_at": completion_time,
        }

    except ReportJobDeterministicError as e:
        error_msg = str(e)
        logger.error(
            f"[GenerateReportActivity] Deterministic failure for job {job_id}: {error_msg}"
        )
        return {
            "job_id": job_id,
            "organization_id": organization_id,
            "status": "FAILED",
            "error": error_msg,
            "error_type": "deterministic",
        }

    except ReportJobTransientError as e:
        error_msg = str(e)
        logger.error(
            f"[GenerateReportActivity] Transient failure for job {job_id}: {error_msg}\n"
            f"Traceback: {traceback.format_exc()}"
        )
        raise

    except asyncio.TimeoutError:
        error_msg = f"Job {job_id} timed out after {JOB_TIMEOUT_SECONDS} seconds"
        logger.error(f"[GenerateReportActivity] {error_msg}")

        # Update job status in Cosmos DB
        error_payload = {
            "error_type": "timeout",
            "error_message": error_msg,
            "attempt": attempt,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        update_report_job_status(
            job_id, organization_id, "FAILED", error_payload=error_payload
        )
        raise

    except Exception as e:
        error_msg = str(e)
        logger.error(
            f"[GenerateReportActivity] Error for job {job_id}: {error_msg}\n"
            f"Traceback: {traceback.format_exc()}"
        )

        # Update job status in Cosmos DB
        error_payload = {
            "error_type": "unexpected",
            "error_message": error_msg,
            "attempt": attempt,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        update_report_job_status(
            job_id, organization_id, "FAILED", error_payload=error_payload
        )
        raise
