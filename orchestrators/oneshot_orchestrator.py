from function_app import app
import azure.durable_functions as df


@app.orchestration_trigger(context_name="ctx")
@app.function_name("OneShotOrchestrator")
def oneshot_orchestrator(ctx: df.DurableOrchestrationContext):
    """
    Simple orchestrator for testing a single job.
    """
    job = ctx.get_input() or {}
    result = yield ctx.call_activity("GenerateReportActivity", job)
    return result
