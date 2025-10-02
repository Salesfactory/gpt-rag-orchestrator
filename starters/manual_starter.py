from function_app import app
import azure.functions as func
import azure.durable_functions as df
import json


@app.route(route="start-orch", methods=[func.HttpMethod.POST])
@app.durable_client_input(client_name="client")
async def start_orch(req: func.HttpRequest, client: df.DurableOrchestrationClient):
    body = req.get_json()
    orch = body.get("orchestrator", "OneShotOrchestrator")
    payload = body.get("input", {})
    instance_id = await client.start_new(orch, client_input=payload)
    return func.HttpResponse(json.dumps({"instanceId": instance_id}), headers={"Content-Type": "application/json"})
