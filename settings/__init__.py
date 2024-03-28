import logging
import azure.functions as func
import json
import os

from shared.util import get_settings, set_settings, get_setting

LOGLEVEL = os.environ.get('LOGLEVEL', 'DEBUG').upper()
logging.basicConfig(level=LOGLEVEL)

async def main(req: func.HttpRequest) -> func.HttpResponse:

    # settings = get_settings()
    if req.method == "POST":
        try:
            req_body = json.loads(req.get_body())
        except:
            return func.HttpResponse("Invalid request body", status_code=400)
        
        logging.info('Python HTTP trigger function processed a request to set settings. request body: %s', req.get_json())
        
        set_settings(
            user_id=req_body.get('user_id','124a5b7c-8d9e-0f1g-2h3i-4j5k6l7m8n9o'), # test uuid, find a way to get this
            temperature=req_body.get('temperature', 0.0),           # adjust default values
            frequency_penalty=req_body.get('frequency_penalty', 0.0), # adjust default values
            presence_penalty=req_body.get('presence_penalty', 0.0),   # adjust default values
        )

        return func.HttpResponse(json.dumps(req_body), mimetype="application/json", status_code=200)
    else:
        logging.info('Python HTTP trigger function processed a request to get settings.')
        
        user_id = req.params.get('user_id')

        if user_id:
            logging.info('User ID found in request body. Getting settings for user: {user_id}')

            settings = get_setting(user_id)
            return func.HttpResponse(json.dumps(settings), mimetype="application/json", status_code=200)
        else: 
            # this may be disabled..
            logging.info('No user ID found in request body. Getting all settings.')
            settings = get_settings()
            return func.HttpResponse(json.dumps(settings), mimetype="application/json", status_code=200)