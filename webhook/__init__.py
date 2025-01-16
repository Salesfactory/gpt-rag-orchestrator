import logging
import azure.functions as func
import json
import os

import stripe


from shared.util import handle_subscription_logs, update_organization_subscription, disable_organization_active_subscription, enable_organization_subscription, update_subscription_logs

LOGLEVEL = os.environ.get("LOGLEVEL", "DEBUG").upper()
logging.basicConfig(level=LOGLEVEL)


async def main(req: func.HttpRequest) -> func.HttpResponse:
    logging.info("Python HTTP trigger function processed a request.")
    if req.method != "POST":
        return func.HttpResponse("Method not allowed", status_code=405)
    
    stripe.api_key = os.getenv("STRIPE_API_KEY")
    endpoint_secret = os.getenv("STRIPE_SIGNING_SECRET")
    stripe_product_fa=os.getenv("STRIPE_PRODUCT_FA")

    event = None
    payload = req.get_body()

    try:
        event = json.loads(payload)
    except json.decoder.JSONDecodeError as e:
        print("  Webhook error while parsing basic request." + str(e))
        return json.dumps({"success": False}), 400
    if endpoint_secret:
        # Only verify the event if there is an endpoint secret defined
        # Otherwise use the basic event deserialized with json
        sig_header = req.headers["stripe-signature"]
        
        try:
            event = stripe.Webhook.construct_event(payload, sig_header, endpoint_secret)
        except stripe.error.SignatureVerificationError as e:
            print("  Webhook signature verification failed. " + str(e))
            return json.dumps({"success": False}), 400

    # Handle the event
    if event["type"] == "checkout.session.completed":
        print("  Webhook received!", event["type"])
        userId = event["data"]["object"]["client_reference_id"]
        organizationId = event["data"]["object"].get("metadata", {}).get("organizationId", "") or ""
        sessionId = event["data"]["object"]["id"]
        subscriptionId = event["data"]["object"]["subscription"]
        paymentStatus = event["data"]["object"]["payment_status"]
        custom_fields = event["data"]["object"].get("custom_fields", [])
        organizationName = custom_fields[0]["text"]["value"] if custom_fields else "No name"
        expirationDate = event["data"]["object"]["expires_at"]
        try:
            update_organization_subscription(userId, organizationId, subscriptionId, sessionId, paymentStatus, organizationName, expirationDate)
            logging.info(f"User {userId} updated with subscription {subscriptionId}")
        except Exception as e:
            logging.exception("[webbackend] exception in /api/webhook")
            return func.HttpResponse(
                json.dumps({"error": f"Error in webhook execution: {str(e)}"}),
                mimetype="application/json",
                status_code=500,
            )
    elif event["type"] == "customer.subscription.updated":
        print("  Webhook received!", event["type"])
        subscriptionId = event["data"]["object"]["id"]
        status = event["data"]["object"]["status"]
        print(event)
        print(f"Subscription {subscriptionId} updated to status {status}")

        def determine_action(event):
            data = event.get("data", {}).get("object", {})
            previous_data = event.get("data", {}).get("previous_attributes", {})
            current_items = data.get("items", {}).get("data", [])
            previous_items = previous_data.get("items", {}).get("data", [])

            #Detect when the subscription is new
            if "status" in previous_data and previous_data["status"] == "incomplete" and data["status"] == "active":
                current_plan = data.get("plan", {})
                current_plan_nickname = current_plan.get("nickname", "Unknown")
                return "New Subscription", None, current_plan_nickname


            # Detect Financial Assistant activated
            if len(previous_items) < len(current_items):
                new_item_ids = {item["id"] for item in current_items} - {item["id"] for item in previous_items}
                for item in current_items:
                    if item["id"] in new_item_ids:
                        if item["price"]["product"] == stripe_product_fa:  # ID del Financial Assistant
                            return "Financial Assistant on", None, None

            # Detect Financial Assistant disabled
            if len(previous_items) > len(current_items):
                removed_item_ids = {item["id"] for item in previous_items} - {item["id"] for item in current_items}
                for item in previous_items:
                    if item["id"] in removed_item_ids:
                        if item["price"]["product"] == stripe_product_fa:  # ID del Financial Assistant
                            return "Financial Assistant off", None, None

            # Detect subscription level change
            if "plan" in previous_data:
                previous_plan = previous_data.get("plan",{}).get("nickname",None)
                current_plan = data.get("plan",{}).get("nickname",None)
                return "Subscription Tier Change",previous_plan,current_plan
            
            # Unknown action
            return "Unknown action", None, None
        
        action, previous_plan, current_plan = determine_action(event)
        print(f"Action determined: {action}")

        if action =="Subscription Tier Change":
            print(f"Previous Plan: {previous_plan}")
            print(f"Current Plan: {current_plan}")

        enable_organization_subscription(subscriptionId)
        update_subscription_logs(subscriptionId, action, previous_plan, current_plan)
    elif event["type"] == "customer.subscription.paused":
        print("  Webhook received!", event["type"])
        subscriptionId = event["data"]["object"]["id"]
        event_type = event["type"].split(".")[-1] # Obtain "paused"
        handle_subscription_logs(subscriptionId, event_type)
        disable_organization_active_subscription(subscriptionId)
    elif event["type"] == "customer.subscription.resumed":
        print("  Webhook received!", event["type"])
        event_type = event["type"].split(".")[-1] # Obtain "resumed"
        handle_subscription_logs(subscriptionId, event_type)
        enable_organization_subscription(subscriptionId)
    elif event["type"] == "customer.subscription.deleted":
        print("  Webhook received!", event["type"])
        event_type = event["type"].split(".")[-1] # Obtain "deleted"
        subscriptionId = event["data"]["object"]["id"]
        handle_subscription_logs(subscriptionId, event_type)
        disable_organization_active_subscription(subscriptionId)
    else:
        # Unexpected event type
        logging.info(f"Unexpected event type: {event['type']}")

    return func.HttpResponse(
        json.dumps({"success": True}), status_code=200
    )
