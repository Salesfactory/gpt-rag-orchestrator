from function_app import app
import azure.durable_functions as df
from collections import defaultdict

WAVE_SIZE = 5  # tune to your comfort level


@app.orchestration_trigger(context_name="ctx")
@app.function_name("MainOrchestrator")
def main_orchestrator(ctx: df.DurableOrchestrationContext):
    """
    Groups jobs by tenant and kicks TenantOrchestrator in bounded waves.
    Input: [ {job with tenant_id, organization_id, job_id, etag}, ...]
    """
    all_jobs = ctx.get_input() or []

    grouped = defaultdict(list)
    for j in all_jobs:
        grouped[j["tenant_id"]].append(j)

    results = []
    wave = []
    for tenant_id, jobs in grouped.items():
        wave.append(ctx.call_sub_orchestrator("TenantOrchestrator", {"tenant_id": tenant_id, "jobs": jobs}))
        if len(wave) >= WAVE_SIZE:
            batch_res = yield ctx.task_all(wave)
            results.extend(batch_res)
            wave = []

    if wave:
        batch_res = yield ctx.task_all(wave)
        results.extend(batch_res)

    return results
