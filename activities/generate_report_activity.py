from function_app import app
import logging
import traceback
from datetime import datetime, timezone

# reuse your existing job logic
from report_worker.processor import process_report_job
from shared.util import update_report_job_status
from shared.cosmos_jobs import (
    cosmos_container,
    try_mark_job_running,
    mark_job_result
)
from shared.observability import log_event


@app.activity_trigger(input_name="job")
@app.function_name("GenerateReportActivity")
async def generate_report_activity(job: dict) -> dict:
    """
    Durable Activity that executes ONE report job with idempotency.
    Expects job to include: job_id, organization_id, tenant_id, etag (Cosmos _etag)
    """
    job_id = job["job_id"]
    org_id = job["organization_id"]
    tenant_id = job.get("tenant_id")
    etag = job.get("etag")
    attempt = job.get("attempt", 1)

    try:
        log_event("ReportJobStart", job_id=job_id, tenant_id=tenant_id, attempt=attempt)

        # Idempotency: try to transition QUEUED -> RUNNING with ETag
        container = cosmos_container()
        if etag and not try_mark_job_running(container, job_id, etag):
            logging.info(f"[GenerateReportActivity] Skip {job_id}, another worker took it.")
            return {"job_id": job_id, "status": "SKIPPED"}

        # Do the actual work (your existing logic)
        await process_report_job(job_id, org_id, attempt)

        # Mark success
        mark_job_result(container, job_id, status="SUCCEEDED")
        log_event("ReportJobCompleted", job_id=job_id, tenant_id=tenant_id, status="SUCCEEDED")
        return {"job_id": job_id, "status": "SUCCEEDED"}

    except Exception as e:
        logging.error(f"[GenerateReportActivity] Error {job_id}: {e}\n{traceback.format_exc()}")
        update_report_job_status(job_id, org_id, "FAILED", {"error": str(e), "attempt": attempt})
        mark_job_result(cosmos_container(), job_id, status="FAILED", error=str(e))
        log_event("ReportJobCompleted", job_id=job_id, tenant_id=tenant_id, status="FAILED", error=str(e))
        # return failure (Durable retry is configured in orchestrator)
        return {"job_id": job_id, "status": "FAILED", "error": str(e)}
