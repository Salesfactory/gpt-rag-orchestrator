import azure.functions as func
import logging
import json
import os
from datetime import datetime, timezone, timedelta
from typing import List

from azurefunctions.extensions.http.fastapi import Request, StreamingResponse, Response
from scheduler.create_batch_jobs import create_batch_jobs

from shared.util import (
    get_user,
    trigger_indexer_with_retry,
)

from orc import ConversationOrchestrator, get_settings
from shared.conversation_export import export_conversation
from webscrapping.multipage_scrape import crawl_website
from report_worker.processor import process_report_job, ReportJobDeterministicError
from shared.cosmos_jobs import (
    load_scheduled_jobs,
    cosmos_container,
    try_mark_job_running,
    acquire_global_lease,
    release_global_lease,
    is_global_lease_active,
    reset_stale_running_jobs,
)
from shared.queue_utils import enqueue_message

# MULTIPAGE SCRAPING CONSTANTS
DEFAULT_LIMIT = 30
DEFAULT_MAX_DEPTH = 4
DEFAULT_MAX_BREADTH = 15

REPORT_SCHEDULE_CRON = os.getenv("REPORT_SCHEDULE_CRON", "0 0 14 * * *")
HOST_INSTANCE_ID = os.getenv("WEBSITE_INSTANCE_ID", "local")

app = func.FunctionApp(http_auth_level=func.AuthLevel.FUNCTION)


@app.function_name(name="report_queue_worker")
@app.queue_trigger(
    arg_name="msg",
    queue_name="report-processing",
    connection="AzureWebJobsStorage",
)
async def report_queue_worker(msg: func.QueueMessage) -> None:
    """
    Queue worker for report generation.
    Processes one report job at a time using a global Cosmos lease.
    """
    try:
        payload = json.loads(msg.get_body().decode("utf-8"))
    except Exception as e:
        logging.error(f"[report-queue-worker] Invalid message body: {e}")
        return

    job_id = payload.get("job_id")
    organization_id = payload.get("organization_id")
    etag = payload.get("etag")
    dequeue_count = msg.dequeue_count or 1

    if not job_id or not organization_id:
        logging.error("[report-queue-worker] Missing job_id or organization_id")
        return

    container = cosmos_container()
    acquired = acquire_global_lease(
        container,
        instance_id=HOST_INSTANCE_ID,
        ttl_minutes=45,
    )
    if not acquired:
        enqueue_message("report-processing", payload, visibility_timeout=90)
        return

    try:
        if etag and dequeue_count == 1:
            claimed = try_mark_job_running(
                container,
                job_id,
                organization_id,
                etag,
                processing_instance_id=HOST_INSTANCE_ID,
            )
            if not claimed:
                logging.info(
                    f"[report-queue-worker] Job {job_id} already claimed, skipping"
                )
                return

        await process_report_job(
            job_id,
            organization_id,
            dequeue_count,
            allow_retry=True,
        )
    except ReportJobDeterministicError as e:
        logging.error(
            f"[report-queue-worker] Deterministic error for job {job_id}: {e}"
        )
        return
    except Exception as e:
        logging.error(f"[report-queue-worker] Transient error: {e}")
        raise
    finally:
        release_global_lease(container, instance_id=HOST_INSTANCE_ID)


@app.function_name(name="report_stale_cleanup")
@app.timer_trigger(
    schedule="0 30 * * * *", arg_name="cleanup_timer", run_on_startup=False
)
async def report_stale_cleanup(cleanup_timer: func.TimerRequest) -> None:
    """
    Reset stale RUNNING jobs to QUEUED when no active global lease exists.
    """
    container = cosmos_container()

    if is_global_lease_active(container):
        logging.info("[report-stale-cleanup] Global lease active; skipping cleanup")
        return

    cutoff = datetime.now(timezone.utc) - timedelta(minutes=45)
    reset_jobs = reset_stale_running_jobs(container, cutoff.isoformat())
    for job in reset_jobs:
        payload = {
            "job_id": job.get("job_id"),
            "organization_id": job.get("organization_id"),
            "tenant_id": job.get("tenant_id"),
            "etag": job.get("etag"),
            "enqueued_at": datetime.now(timezone.utc).isoformat(),
        }
        enqueue_message("report-processing", payload)

    logging.info(f"[report-stale-cleanup] Reset {len(reset_jobs)} stale RUNNING job(s)")


@app.route(
    route="health", methods=[func.HttpMethod.GET], auth_level=func.AuthLevel.ANONYMOUS
)
async def health_check(req: Request) -> Response:
    """
    Health check endpoint for Azure App Service health monitoring.
    pinged by Azure's health check feature at 1-minute intervals

    Returns:
        200 OK when the application is healthy
    """
    return Response("OK", status_code=200, media_type="text/plain")


@app.function_name(name="report_queue_scheduler")
@app.timer_trigger(
    schedule=REPORT_SCHEDULE_CRON, arg_name="mytimer", run_on_startup=False
)
@app.queue_output(
    arg_name="queue_msgs",
    queue_name="report-processing",
    connection="AzureWebJobsStorage",
)
async def report_queue_scheduler(
    mytimer: func.TimerRequest, queue_msgs: func.Out[List[str]]
) -> None:
    logging.info("[report-queue-scheduler] Timer trigger started")

    try:
        batch_result = create_batch_jobs()
        logging.info(
            f"[report-queue-scheduler] Created {batch_result.get('total_created', 0)} jobs"
        )

        jobs = await load_scheduled_jobs()
        if not jobs:
            logging.info("[report-queue-scheduler] No jobs to enqueue")
            return

        messages = []
        for job in jobs:
            msg = {
                "job_id": job["job_id"],
                "organization_id": job["organization_id"],
                "tenant_id": job["tenant_id"],
                "etag": job.get("etag"),
                "enqueued_at": datetime.now(timezone.utc).isoformat(),
            }
            messages.append(json.dumps(msg))

        queue_msgs.set(messages)
        logging.info(f"[report-queue-scheduler] Enqueued {len(messages)} report jobs")
    except Exception as e:
        logging.error(f"[report-queue-scheduler] Failed: {str(e)}")
        raise


@app.route(route="orc", methods=[func.HttpMethod.POST])
async def stream_response(req: Request) -> StreamingResponse:
    """Endpoint to stream LLM responses to the client"""
    logging.info("[orc] Python HTTP trigger function processed a request.")

    req_body = await req.json()
    question = req_body.get("question")
    conversation_id = req_body.get("conversation_id")
    user_timezone = req_body.get("user_timezone")
    blob_names = req_body.get("blob_names", [])
    is_data_analyst_mode = req_body.get("is_data_analyst_mode", False)
    is_agentic_search_mode = req_body.get("is_agentic_search_mode", False)
    client_principal_id = req_body.get("client_principal_id")
    client_principal_name = req_body.get("client_principal_name")
    client_principal_organization = req_body.get("client_principal_organization")
    if not client_principal_id or client_principal_id == "":
        client_principal_id = "00000000-0000-0000-0000-000000000000"
        client_principal_name = "anonymous"
        client_principal_organization = "00000000-0000-0000-0000-000000000000"
    client_principal = {
        "id": client_principal_id,
        "name": client_principal_name,
        "organization": client_principal_organization,
    }

    organization_id = None
    user = get_user(client_principal_id)
    if "data" in user:
        organization_id = client_principal_organization

        logging.info(
            f"[FunctionApp] Retrieved organizationId: {organization_id} from user data"
        )

    # print configuration settings for the user
    settings = get_settings(client_principal)
    logging.info(f"[function_app] Configuration settings: {settings}")

    # validate settings
    temp_setting = settings.get("temperature")
    settings["temperature"] = float(temp_setting) if temp_setting is not None else 0.3
    settings["model"] = settings.get("model") or "gpt-4.1"
    logging.info(f"[function_app] Validated settings: {settings}")
    if question:
        orchestrator = ConversationOrchestrator(organization_id=organization_id)
        try:
            logging.info("[FunctionApp] Processing conversation")
            return StreamingResponse(
                orchestrator.generate_response_with_progress(
                    conversation_id=conversation_id,
                    question=question,
                    user_info=client_principal,
                    user_settings=settings,
                    user_timezone=user_timezone,
                    blob_names=blob_names,
                    is_data_analyst_mode=is_data_analyst_mode,
                    is_agentic_search_mode=is_agentic_search_mode,
                ),
                media_type="text/event-stream",
            )
        except Exception as e:
            logging.error(f"[FunctionApp] Error in progress streaming: {str(e)}")
            return StreamingResponse(
                '{"error": "error in response generation"}',
                media_type="application/json",
            )
    else:
        return StreamingResponse(
            '{"error": "no question found in json input"}',
            media_type="application/json",
        )


@app.function_name(name="EventGridTrigger")
@app.event_grid_trigger(arg_name="event")
def blob_event_grid_trigger(event: func.EventGridEvent):
    """
    Event Grid trigger that triggers the search indexer when blob events are received.
    Filtering is handled at the infrastructure level.
    Supports multiple indexers separated by commas.
    """
    try:
        index_names = os.getenv("AZURE_AI_SEARCH_INDEX_NAME", "")
        if not index_names:
            logging.warning(
                "[blob_event_grid] AZURE_AI_SEARCH_INDEX_NAME not configured"
            )
            return

        # Split by comma and strip whitespace to support multiple indexes
        index_list = [name.strip() for name in index_names.split(",") if name.strip()]

        logging.info(
            f"[blob_event_grid] Event received for blob: {event.subject}, triggering {len(index_list)} indexer(s)"
        )

        for index_name in index_list:
            # Handle special case: pulse-index uses pulse-indexer (not pulse-index-indexer)
            if index_name == "pulse-index":
                indexer_name = "pulse-indexer"
            else:
                indexer_name = f"{index_name}-indexer"
            logging.info(f"[blob_event_grid] Triggering indexer '{indexer_name}'")

            indexer_success = trigger_indexer_with_retry(indexer_name, event.subject)

            if indexer_success:
                logging.info(
                    f"[blob_event_grid] Successfully triggered indexer '{indexer_name}'"
                )
            else:
                logging.warning(
                    f"[blob_event_grid] Could not trigger indexer '{indexer_name}'"
                )

    except Exception as e:
        logging.error(f"[blob_event_grid] Error: {str(e)}, Event ID: {event.id}")


@app.route(
    route="conversations",
    methods=[func.HttpMethod.POST],
)
async def conversations(req: Request) -> Response:
    logging.info("Python HTTP trigger function processed a request for conversations.")

    if req.method == "POST":
        try:
            req_body = await req.json()
            id_from_body = req_body.get("id")
            if not id_from_body:
                return Response("Missing conversation ID for export", status_code=400)

            user_id = req_body.get("user_id")
            export_format = req_body.get("format", "html")

            if not user_id:
                return Response("Missing user_id in request body", status_code=400)

            if export_format not in ["html", "json"]:
                return Response(
                    "Invalid export format. Supported formats: html, json",
                    status_code=400,
                )

            result = export_conversation(id_from_body, user_id, export_format)

            if result["success"]:
                return Response(
                    json.dumps(result), media_type="application/json", status_code=200
                )
            else:
                return Response(
                    json.dumps({"error": result["error"]}),
                    media_type="application/json",
                    status_code=500,
                )

        except json.JSONDecodeError:
            return Response("Invalid JSON in request body", status_code=400)
        except Exception as e:
            logging.error(f"Error in conversation export: {str(e)}")
            return Response(
                json.dumps({"error": "Internal server error"}),
                media_type="application/json",
                status_code=500,
            )
    else:
        return Response("Method not allowed", status_code=405)


@app.route(route="scrape-page", methods=[func.HttpMethod.POST])
async def scrape_page(req: Request) -> Response:
    """
    Endpoint to scrape a single web page.

    Expected payload:
    {
        "url": "http://example.com",
        "client_principal_id": "user-id"
    }

    Returns:
        JSON response with scraping results and optional blob storage results
    """
    logging.info("[scrape-pages] Python HTTP trigger function processed a request.")

    try:

        req_body = await req.json()

        # Validate payload
        if not req_body or "url" not in req_body:
            return Response(
                content=json.dumps(
                    {
                        "status": "error",
                        "message": "Request body must contain 'url' field",
                    }
                ),
                media_type="application/json",
                status_code=400,
            )

        url = req_body["url"]
        if not isinstance(url, str) or not url.strip():
            return Response(
                content=json.dumps(
                    {"status": "error", "message": "url must be a non-empty string"}
                ),
                media_type="application/json",
                status_code=400,
            )

        # Extract client principal ID and organization
        client_principal_id = req_body.get(
            "client_principal_id", "00000000-0000-0000-0000-000000000000"
        )

        organization_id = None
        try:
            user = get_user(client_principal_id)
            organization_id = user.get("data", {}).get("organizationId")
            if organization_id:
                logging.info(
                    f"[scrape-pages] Retrieved organizationId: {organization_id}"
                )
        except Exception as e:
            logging.info(f"[scrape-pages] No organization tracking - {str(e)}")

        from webscrapping import scrape_single_url
        from webscrapping.utils import generate_request_id

        request_id = req.headers.get("x-request-id") or generate_request_id()

        result_data = scrape_single_url(url.strip(), request_id, organization_id)

        result_status = result_data.get("status")
        if result_status == "completed":
            status_code = 200
        elif result_status == "failed":
            status_code = 422
        else:
            status_code = 500

        return Response(
            content=json.dumps(result_data),
            media_type="application/json",
            status_code=status_code,
        )

    except json.JSONDecodeError:
        return Response(
            content=json.dumps({"status": "error", "message": "Invalid JSON format"}),
            media_type="application/json",
            status_code=400,
        )
    except Exception as e:
        logging.error(f"Error in scrape-pages endpoint: {str(e)}")
        return Response(
            content=json.dumps(
                {"status": "error", "message": f"Internal server error: {str(e)}"}
            ),
            media_type="application/json",
            status_code=500,
        )


def create_preview_results(results: list, preview_length: int = 100) -> list:
    """
    Create a preview version of crawl results with truncated raw_content.

    Args:
        results: List of crawl results from Tavily
        preview_length: Number of characters to show in preview (default: 100)

    Returns:
        List of results with truncated raw_content for API response
    """
    if not results:
        return results

    preview_results = []
    for result in results:
        # Create a copy of the result
        preview_result = result.copy()

        # Truncate raw_content if it exists
        if "raw_content" in preview_result and preview_result["raw_content"]:
            content = preview_result["raw_content"]
            if len(content) > preview_length:
                preview_result["raw_content"] = content[:preview_length] + "..."

        preview_results.append(preview_result)

    return preview_results


@app.route(route="multipage-scrape", methods=[func.HttpMethod.POST])
async def multipage_scrape(req: Request) -> Response:
    """
    Endpoint to crawl a website using advanced multipage scraping with Tavily.

    Expected payload:
    {
        "url": "https://example.com",
        "limit": 30,           // optional, default 30
        "max_depth": 4,        // optional, default 4
        "max_breadth": 15,     // optional, default 15
        "client_principal_id": "user-id"  // optional
    }

    Returns:
        JSON response with crawling results including all discovered pages
    """
    logging.info("[multipage-scrape] Python HTTP trigger function processed a request.")

    try:
        req_body = await req.json()

        # Validate payload
        if not req_body or "url" not in req_body:
            return Response(
                content=json.dumps(
                    {
                        "status": "error",
                        "message": "Request body must contain 'url' field",
                    }
                ),
                media_type="application/json",
                status_code=400,
            )

        url = req_body["url"]
        if not url or not isinstance(url, str):
            return Response(
                content=json.dumps(
                    {"status": "error", "message": "url must be a non-empty string"}
                ),
                media_type="application/json",
                status_code=400,
            )

        limit = req_body.get("limit", DEFAULT_LIMIT)
        max_depth = req_body.get("max_depth", DEFAULT_MAX_DEPTH)
        max_breadth = req_body.get("max_breadth", DEFAULT_MAX_BREADTH)

        if not isinstance(limit, int) or limit < 1 or limit > 100:
            return Response(
                content=json.dumps(
                    {
                        "status": "error",
                        "message": "limit must be an integer between 1 and 100",
                    }
                ),
                media_type="application/json",
                status_code=400,
            )

        if not isinstance(max_depth, int) or max_depth < 1 or max_depth > 10:
            return Response(
                content=json.dumps(
                    {
                        "status": "error",
                        "message": "max_depth must be an integer between 1 and 10",
                    }
                ),
                media_type="application/json",
                status_code=400,
            )

        if not isinstance(max_breadth, int) or max_breadth < 1 or max_breadth > 50:
            return Response(
                content=json.dumps(
                    {
                        "status": "error",
                        "message": "max_breadth must be an integer between 1 and 50",
                    }
                ),
                media_type="application/json",
                status_code=400,
            )

        # Extract client principal ID for logging/tracking
        client_principal_id = req_body.get(
            "client_principal_id", "00000000-0000-0000-0000-000000000000"
        )

        organization_id = None
        try:
            user = get_user(client_principal_id)
            organization_id = user.get("data", {}).get("organizationId")
            if organization_id:
                logging.info(
                    f"[multipage-scrape] Retrieved organizationId: {organization_id}"
                )
        except Exception as e:
            logging.info(f"[multipage-scrape] No organization tracking - {str(e)}")

        logging.info(
            f"[multipage-scrape] Starting crawl for URL: {url} with limit: {limit}, max_depth: {max_depth}, max_breadth: {max_breadth}"
        )

        # Extract request ID from headers if provided, or generate one
        from webscrapping.utils import generate_request_id

        request_id = req.headers.get("x-request-id") or generate_request_id()

        # Execute the multipage crawling
        crawl_result = crawl_website(url, limit, max_depth, max_breadth)

        # Check if crawling was successful
        if "error" in crawl_result:
            return Response(
                content=json.dumps(
                    {
                        "status": "error",
                        "message": f"Crawling failed: {crawl_result['error']}",
                        "url": url,
                    }
                ),
                media_type="application/json",
                status_code=500,
            )

        # Initialize blob storage (always enabled)
        from webscrapping.blob_manager import create_crawler_manager_from_env
        from webscrapping.scraper import WebScraper

        crawler_manager = create_crawler_manager_from_env(request_id)
        blob_storage_result = None

        # Handle blob storage for all successful crawls
        if crawl_result.get("results"):
            # Format crawl results for blob storage
            crawl_parameters = {
                "limit": limit,
                "max_depth": max_depth,
                "max_breadth": max_breadth,
            }

            formatted_pages = WebScraper.format_multipage_content_for_blob_storage(
                crawl_result=crawl_result,
                request_id=request_id,
                organization_id=organization_id,
                original_url=url,
                crawl_parameters=crawl_parameters,
            )

            if crawler_manager and formatted_pages:
                try:
                    # Upload to blob storage
                    blob_storage_result = (
                        crawler_manager.store_multipage_results_in_blob(
                            formatted_pages=formatted_pages, content_type="text/plain"
                        )
                    )

                    logging.info(
                        f"[multipage-scrape] Blob storage: {blob_storage_result['total_successful']} uploaded, "
                        f"{blob_storage_result['total_failed']} failed, {blob_storage_result['total_duplicates']} duplicates"
                    )

                except Exception as blob_error:
                    blob_storage_result = {
                        "status": "error",
                        "error": f"Blob storage upload failed: {str(blob_error)}",
                        "total_processed": len(formatted_pages),
                        "total_successful": 0,
                        "total_failed": len(formatted_pages),
                        "total_duplicates": 0,
                    }
                    logging.error(
                        f"[multipage-scrape] Blob storage failed for URL: {url}, error: {str(blob_error)}"
                    )
            elif not crawler_manager:
                # Storage not configured
                blob_storage_result = {
                    "status": "not_configured",
                    "message": "Blob storage not configured - missing Azure storage environment variables",
                    "total_processed": len(formatted_pages) if formatted_pages else 0,
                    "total_successful": 0,
                    "total_failed": 0,
                    "total_duplicates": 0,
                }
                logging.info(
                    f"[multipage-scrape] Blob storage not configured for URL: {url}"
                )
            else:
                # Failed to format pages
                blob_storage_result = {
                    "status": "error",
                    "error": "Failed to format pages for blob storage",
                    "total_processed": 0,
                    "total_successful": 0,
                    "total_failed": 0,
                    "total_duplicates": 0,
                }
        else:
            # No results to store
            blob_storage_result = {
                "status": "no_content",
                "message": "No pages found to store",
                "total_processed": 0,
                "total_successful": 0,
                "total_failed": 0,
                "total_duplicates": 0,
            }

        # Create preview results for API response (truncated raw_content)
        preview_results = create_preview_results(crawl_result.get("results", []))

        # Generate message based on blob storage result
        if blob_storage_result.get("total_successful", 0) > 0:
            if blob_storage_result.get("total_failed", 0) > 0:
                message = f"Scraped {blob_storage_result['total_successful']} pages successfully, {blob_storage_result['total_failed']} failed"
            else:
                message = f"Successfully scraped {blob_storage_result['total_successful']} pages and uploaded to blob storage"
        elif blob_storage_result.get("status") == "not_configured":
            message = f"Scraped {len(crawl_result.get('results', []))} pages (blob storage not configured)"
        else:
            message = f"Scraped {len(crawl_result.get('results', []))} pages but blob storage failed"

        # Format successful response with preview results
        response_data = {
            "status": "completed",
            "message": message,
            "url": url,
            "parameters": {
                "limit": limit,
                "max_depth": max_depth,
                "max_breadth": max_breadth,
            },
            "results": preview_results,
            "pages_found": len(crawl_result.get("results", [])),
            "response_time": crawl_result.get("response_time", 0.0),
            "organization_id": organization_id,
            "request_id": request_id,
            "blob_storage_result": blob_storage_result,
        }

        logging.info(
            f"[multipage-scrape] Successfully crawled {response_data['pages_found']} pages from {url}"
        )

        return Response(
            content=json.dumps(response_data),
            media_type="application/json",
            status_code=200,
        )

    except json.JSONDecodeError:
        return Response(
            content=json.dumps({"status": "error", "message": "Invalid JSON format"}),
            media_type="application/json",
            status_code=400,
        )
    except Exception as e:
        logging.error(f"Error in multipage-scrape endpoint: {str(e)}")
        return Response(
            content=json.dumps(
                {"status": "error", "message": f"Internal server error: {str(e)}"}
            ),
            media_type="application/json",
            status_code=500,
        )
