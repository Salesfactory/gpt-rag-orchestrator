from durable_functions_registry import app
import azure.durable_functions as df
from datetime import timedelta
import logging


@app.orchestration_trigger(context_name="context")
def TenantOrchestrator(context: df.DurableOrchestrationContext):
    """
    Runs a list of jobs for ONE tenant, sequentially (or you can extend to small batches).
    Expects input: {"tenant_id": "...", "jobs": [ {job}, ... ]}
    """
    payload = context.get_input() or {}
    tenant_id = payload["tenant_id"]
    jobs = list(payload["jobs"])

    limiter = df.EntityId("RateLimiter", "global")
    results = []
    retry_opts = df.RetryOptions(
        first_retry_interval_in_milliseconds=30000, max_number_of_attempts=2
    )

    for job in jobs:
        max_acquire_attempts = 3
        acquire_attempts = 0
        granted = False

        while acquire_attempts < max_acquire_attempts:
            grant = yield context.call_entity(
                limiter, "acquire", {"tenant_id": tenant_id}
            )
            if grant.get("granted"):
                granted = True
                break
            acquire_attempts += 1
            wait_ms = grant.get("wait_ms", 1500)
            yield context.create_timer(
                context.current_utc_datetime + timedelta(milliseconds=wait_ms)
            )

        if not granted:
            logging.error(
                f"[TenantOrchestrator] Failed to acquire capacity for job {job.get('job_id')} after {max_acquire_attempts} attempts"
            )
            results.append({
                "job_id": job.get("job_id"),
                "status": "FAILED",
                "error": "Failed to acquire rate limiter capacity - timeout"
            })
            continue  # Skip to next job - no token acquired, nothing to release

        # Token acquired - must release after processing
        try:
            job_payload = dict(job)
            job_payload["processing_instance_id"] = context.instance_id
            res = yield context.call_activity_with_retry(
                "GenerateReportActivity", retry_opts, job_payload
            )
            results.append(res)
        except Exception as e:
            # capture failures and continue
            logging.error(f"[TenantOrchestrator] job failed: {e}")
            results.append(
                {"job_id": job.get("job_id"), "status": "FAILED", "error": str(e)}
            )

        # Release token after processing (runs for both success and failure)
        yield context.call_entity(limiter, "release", {"tenant_id": tenant_id})

    return results
