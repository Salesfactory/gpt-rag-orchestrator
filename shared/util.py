# utility functions

import re
import json
import logging
import os
import requests
import tiktoken
import time
import urllib.parse
import uuid
from azure.cosmos import CosmosClient
from azure.cosmos.partition_key import PartitionKey 
from azure.keyvault.secrets import SecretClient
from azure.identity import DefaultAzureCredential
from tenacity import retry, wait_random_exponential, stop_after_attempt
import semantic_kernel as sk
from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion

# logging level
logging.getLogger('azure').setLevel(logging.WARNING)
LOGLEVEL = os.environ.get('LOGLEVEL', 'DEBUG').upper()
logging.basicConfig(level=LOGLEVEL)

# Env variables
AZURE_OPENAI_TEMPERATURE = os.environ.get("AZURE_OPENAI_TEMPERATURE") or "0.17"
AZURE_OPENAI_TOP_P = os.environ.get("AZURE_OPENAI_TOP_P") or "0.27"
AZURE_OPENAI_RESP_MAX_TOKENS = os.environ.get("AZURE_OPENAI_MAX_TOKENS") or "1000"
AZURE_OPENAI_LOAD_BALANCING = os.environ.get("AZURE_OPENAI_LOAD_BALANCING") or "false"
AZURE_OPENAI_LOAD_BALANCING = True if AZURE_OPENAI_LOAD_BALANCING.lower() == "true" else False
AZURE_OPENAI_CHATGPT_MODEL = os.environ.get("AZURE_OPENAI_CHATGPT_MODEL")
AZURE_OPENAI_EMBEDDING_MODEL = os.environ.get("AZURE_OPENAI_EMBEDDING_MODEL")
ORCHESTRATOR_MESSAGES_LANGUAGE = os.environ.get("ORCHESTRATOR_MESSAGES_LANGUAGE") or "en"
AZURE_DB_ID = os.environ.get("AZURE_DB_ID")
AZURE_DB_NAME = os.environ.get("AZURE_DB_NAME")
AZURE_DB_URI = f"https://{AZURE_DB_ID}.documents.azure.com:443/"

model_max_tokens = {
    'gpt-35-turbo': 4096,
    'gpt-35-turbo-16k': 16384,
    'gpt-4': 8192,
    'gpt-4-32k': 32768
}

##########################################################
# KEY VAULT 
##########################################################

def get_secret(secretName):
    start_time = time.time()
    keyVaultName = os.environ["AZURE_KEY_VAULT_NAME"]
    KVUri = f"https://{keyVaultName}.vault.azure.net"
    credential = DefaultAzureCredential()
    client = SecretClient(vault_url=KVUri, credential=credential)
    retrieved_secret = client.get_secret(secretName)
    round(time.time() - start_time,2)
    logging.info(f"[util__module] get_secret: retrieving {secretName} secret from {keyVaultName}.")   
    return retrieved_secret.value

##########################################################
# HISTORY FUNCTIONS
##########################################################

def get_chat_history_as_text(history, include_last_turn=True, approx_max_tokens=1000):
    history_text = ""
    if len(history) == 0:
        return history_text
    for h in reversed(history if include_last_turn else history[:-1]):
        history_text = f"{h['role']}:" + h["content"]  + "\n" + history_text
        if len(history_text) > approx_max_tokens*4:
            break    
    return history_text

def get_chat_history_as_messages(history, include_previous_questions=True, include_last_turn=True, approx_max_tokens=1000):
    history_list = []
    if len(history) == 0:
        return history_list
    for h in reversed(history if include_last_turn else history[:-1]):
        history_item = {"role": h["role"], "content": h["content"]}
        if "function_call" in h:
            history_item.update({"function_call": h["function_call"]})
        if "name" in h:
            history_item.update({"name": h["name"]}) 
        history_list.insert(0, history_item)
        if len(history_list) > approx_max_tokens*4:
            break

    # remove previous questions if needed
    if not include_previous_questions:
        new_list = []
        for idx, item in enumerate(history_list):
            # keep only assistant messages and the last message
            # obs: if include_last_turn is True, the last user message is also kept 
            if item['role'] == 'assistant' or idx == len(history_list)-1:
                new_list.append(item)
        history_list = new_list        
    
    return history_list

##########################################################
# GPT FUNCTIONS
##########################################################

def number_of_tokens(messages, model):
    prompt = json.dumps(messages)
    encoding = tiktoken.encoding_for_model(model.replace('gpt-35-turbo','gpt-3.5-turbo'))
    num_tokens = len(encoding.encode(prompt))
    return num_tokens

def truncate_to_max_tokens(text, extra_tokens, model):
    max_tokens = model_max_tokens[model] - extra_tokens
    tokens_allowed = max_tokens - number_of_tokens(text, model=model)
    while tokens_allowed < int(AZURE_OPENAI_RESP_MAX_TOKENS) and len(text) > 0:
        text = text[:-1]
        tokens_allowed = max_tokens - number_of_tokens(text, model=model)
    return text

# reduce messages to fit in the model's max tokens
def optmize_messages(chat_history_messages, model): 
    messages = chat_history_messages
    # check each get_sources function message and reduce its size to fit into the model's max tokens
    for idx, message in enumerate(messages):
        if message['role'] == 'function' and message['name'] == 'get_sources':
            # top tokens to the max tokens allowed by the model
            sources = json.loads(message['content'])['sources']

            tokens_allowed = model_max_tokens[model] - number_of_tokens(json.dumps(messages), model=model)
            while tokens_allowed < int(AZURE_OPENAI_RESP_MAX_TOKENS) and len(sources) > 0:
                sources = sources[:-1]
                content = json.dumps({"sources": sources})
                messages[idx]['content'] = content                
                tokens_allowed = model_max_tokens[model] - number_of_tokens(json.dumps(messages), model=model)

    return messages
   
@retry(wait=wait_random_exponential(min=20, max=60), stop=stop_after_attempt(6), reraise=True)
async def call_semantic_function(kernel, function, context):
    semantic_response = await kernel.run_async(
        function,
        input_vars=context.variables
    )
    if semantic_response.error_occurred:
        error_code = 'none'
        if hasattr(semantic_response.last_exception, 'error_code'):
            error_code = str(semantic_response.last_exception.error_code)
        error_details = f"Error code: {error_code}. Error message: {semantic_response.last_error_description}"
        logging.info(f"[call_semantic_function] error occurred when calling semantic function {function.name}. {error_details}")
        if error_code == 'ErrorCodes.ServiceError':
            # TODO: add time penalty for model when service is unavailable
            pass
        raise Exception(f"Semantic function {function.name} failed with error: {semantic_response.last_error_description}")
    return semantic_response

@retry(wait=wait_random_exponential(min=2, max=60), stop=stop_after_attempt(6), reraise=True)
def chat_complete(messages, functions, function_call='auto'):
    """  Return assistant chat response based on user query. Assumes existing list of messages """

    oai_config = get_aoai_config(AZURE_OPENAI_CHATGPT_MODEL)

    messages = optmize_messages(messages, AZURE_OPENAI_CHATGPT_MODEL)

    url = f"{oai_config['endpoint']}/openai/deployments/{oai_config['deployment']}/chat/completions?api-version={oai_config['api_version']}"

    headers = {
        "Content-Type": "application/json",
        # "api-key": oai_config['api_key']
        "Authorization": "Bearer "+ oai_config['api_key'] 
    }

    data = {
        "messages": messages,
        "functions": functions,
        "function_call": function_call,
        "max_tokens": int(AZURE_OPENAI_RESP_MAX_TOKENS)
    }

    if function_call == 'auto':
        data['temperature'] = 0
    else:
        data['temperature'] = float(AZURE_OPENAI_TEMPERATURE)
        data['top_p'] = float(AZURE_OPENAI_TOP_P) 

    start_time = time.time()
    response = requests.post(url, headers=headers, data=json.dumps(data)).json()
    response_time =  round(time.time() - start_time,2)
    logging.info(f"[util__module] called chat completion api in {response_time:.6f} seconds")

    return response

##########################################################
# FORMATTING FUNCTIONS
##########################################################

# enforce answer format to the desired format (html, markdown, none)
def format_answer(answer, format= 'none'):
    
    formatted_answer = answer
    
    if format == 'html':
        
        # Convert bold syntax (**text**) to HTML
        formatted_answer = re.sub(r'\*\*(.*?)\*\*', r'<strong>\1</strong>', formatted_answer)
        
        # Convert italic syntax (*text*) to HTML
        formatted_answer = re.sub(r'\*(.*?)\*', r'<em>\1</em>', formatted_answer)
        
        # Return the converted text
    
    elif format == 'markdown':
        formatted_answer = answer # TODO
    
    elif format == 'none':        
        formatted_answer = answer # TODO

    return formatted_answer
  
# replace [doc1] [doc2] [doc3] with the corresponding filepath
def replace_doc_ids_with_filepath(answer, citations):
    for i, citation in enumerate(citations):
        filepath = urllib.parse.quote(citation['filepath'])
        answer = answer.replace(f"[doc{i+1}]", f"[{filepath}]")
    return answer

##########################################################
# MESSAGES FUNCTIONS
##########################################################

def get_message(message):
    if ORCHESTRATOR_MESSAGES_LANGUAGE.startswith("pt"):
        messages_file = "orc/messages/pt.json"
    elif ORCHESTRATOR_MESSAGES_LANGUAGE.startswith("es"):
        messages_file = "orc/messages/es.json"
    else:
        messages_file = "orc/messages/en.json"
    with open(messages_file, 'r') as f:
        json_data = f.read()
    messages_dict = json.loads(json_data)
    return messages_dict[message]

##########################################################
# SEMANTIC KERNEL
##########################################################

def load_sk_plugin(name, oai_config):
    kernel = sk.Kernel()
    kernel.add_chat_service("chat_completion", AzureChatCompletion(oai_config['deployment'], oai_config['endpoint'], oai_config['api_key'], ad_auth=True))
    plugin = kernel.import_semantic_skill_from_directory("orc/plugins", name)
    native_functions = kernel.import_native_skill_from_directory("orc/plugins", name)
    plugin.update(native_functions)
    return plugin


##########################################################
# AOAI FUNCTIONS
##########################################################

def get_list_from_string(string):
    result = string.split(',')
    result = [item.strip() for item in result]
    return result

def get_aoai_config(model):

    resource = get_next_resource(model)
    
    credential = DefaultAzureCredential()
    token = credential.get_token("https://cognitiveservices.azure.com/.default")

    if model in ('gpt-35-turbo', 'gpt-35-turbo-16k', 'gpt-4', 'gpt-4-32k'):
        deployment = os.environ.get("AZURE_OPENAI_CHATGPT_DEPLOYMENT")
    elif model == AZURE_OPENAI_EMBEDDING_MODEL:
        deployment = os.environ.get("AZURE_OPENAI_EMBEDDING_DEPLOYMENT")
    else:
        raise Exception(f"Model {model} not supported. Check if you have the correct env variables set.")

    result ={
        "resource": resource,
        "endpoint": f"https://{resource}.openai.azure.com",
        "deployment": deployment,
        "model": model, # ex: 'gpt-35-turbo-16k', 'gpt-4', 'gpt-4-32k'
        "api_version": os.environ.get("AZURE_OPENAI_API_VERSION") or "2023-08-01-preview",
        "api_key": token.token
    }
    return result

def get_conversations(user_id):
    try:
        credential = DefaultAzureCredential()
        db_client = CosmosClient(AZURE_DB_URI, credential, consistency_level='Session')
        db = db_client.get_database_client(database=AZURE_DB_NAME)
        container = db.get_container_client('conversations')
        conversations = container.query_items(query='SELECT * FROM c WHERE c.conversation_data.interactions[0].user_id  = @user_id', parameters=[dict(name='@user_id', value=user_id)], enable_cross_partition_query=True)
        formatted_conversations = [ {'id': con['id'], 'start_date': con['conversation_data']['start_date'], 'content': con['history'][0]['content']} for con in conversations]
        return formatted_conversations
    except Exception:
        logging.error("Error retrieving the conversations")
        return []

def get_conversation(conversation_id, user_id):
    try: 
        credential = DefaultAzureCredential()
        db_client = CosmosClient(AZURE_DB_URI, credential, consistency_level='Session')
        db = db_client.get_database_client(database=AZURE_DB_NAME)
        container = db.get_container_client('conversations')
        conversation = container.read_item(item=conversation_id, partition_key=conversation_id)
        if conversation['conversation_data']['interactions'][0]['user_id'] != user_id:
            return {}
        return conversation
    except Exception:
        logging.error(f"Error retrieving the conversation '{conversation_id}'")
        return {}

def get_next_resource(model):
    
    # define resource
    resources = os.environ.get("AZURE_OPENAI_RESOURCE")
    resources = get_list_from_string(resources)

    if not AZURE_OPENAI_LOAD_BALANCING or model == AZURE_OPENAI_EMBEDDING_MODEL:
        return resources[0]
    else:
        # get current resource list from cache
        start_time = time.time()
        credential = DefaultAzureCredential()
        db_client = CosmosClient(AZURE_DB_URI, credential, consistency_level='Session')
        db = db_client.get_database_client(database=AZURE_DB_NAME)
        container = db.get_container_client('models')
        try:
            keyvalue = container.read_item(item=model, partition_key=model)
            # check if there's an update in the resource list and update cache
            if set(keyvalue["resources"]) != set(resources):
                keyvalue["resources"] = resources           
        except Exception:
            logging.info(f"[util__module] get_next_resource: first time execution (keyvalue store with '{model}' id does not exist, creating a new one).")  
            keyvalue = { 
                "id": model,
                "resources": resources              
            }      
            keyvalue = container.create_item(body=keyvalue)
        resources= keyvalue["resources"]

        # get the first resource and move it to the end of the list
        resource = resources.pop(0)
        resources.append(resource)

        # update cache
        keyvalue["resources"] = resources
        keyvalue = container.replace_item(item=model, body=keyvalue)

        response_time = round(time.time() - start_time,2)

        logging.info(f"[util__module] get_next_resource: model '{model}' resource {resource}. {response_time} seconds") 
        return resource
    
##########################################################
# OTHER FUNCTIONS
##########################################################

def get_blocked_list():
    blocked_list = []
    credential = DefaultAzureCredential()
    db_client = CosmosClient(AZURE_DB_URI, credential, consistency_level='Session')
    db = db_client.get_database_client(database=AZURE_DB_NAME)
    container = db.get_container_client('guardrails')
    try:
        key_value = container.read_item(item='blocked_list', partition_key='blocked_list')
        blocked_list= key_value["blocked_words"]
        blocked_list = [word.lower() for word in blocked_list]  
    except Exception as e:
        logging.info(f"[util__module] get_blocked_list: no blocked words list (keyvalue store with 'blocked_list' id does not exist).")
    return blocked_list


##########################################################
# SETTINGS
##########################################################

def get_setting(client_principal):
    if not client_principal['id']:
        return {}
    
    logging.info("User ID found. Getting settings for user: " + client_principal['id'])
    
    setting = {}
    credential = DefaultAzureCredential()
    db_client = CosmosClient(AZURE_DB_URI, credential, consistency_level='Session')
    db = db_client.get_database_client(database=AZURE_DB_NAME)
    container = db.get_container_client('settings')
    try:
        query = "SELECT * FROM c WHERE c.user_id = @user_id"
        parameters = [
            {"name": "@user_id", "value": client_principal['id']}
        ]
        result = list(container.query_items(
            query=query,
            parameters=parameters,
            enable_cross_partition_query=True
        ))
        if result:
            setting = result[0]
    except Exception as e:
        logging.info(f"[util__module] get_setting: no settings found for user {client_principal['id']} (keyvalue store with '{client_principal['id']}' id does not exist).")
    return setting


def get_settings():
    settings = []
    credential = DefaultAzureCredential()
    db_client = CosmosClient(AZURE_DB_URI, credential, consistency_level='Session')
    db = db_client.get_database_client(database=AZURE_DB_NAME)
    container = db.get_container_client('settings')
    try:
        settings = container.query_items(query='SELECT * FROM s', enable_cross_partition_query=True)
        settings = list(settings)
        
    except Exception as e:
        logging.info(f"[util__module] get_settings: no settings found (keyvalue store with 'settings' id does not exist).")
    return settings


def set_settings(client_principal, temperature, frequency_penalty, presence_penalty):
    new_setting = {}
    credential = DefaultAzureCredential()
    db_client = CosmosClient(AZURE_DB_URI, credential, consistency_level='Session')
    db = db_client.get_database_client(database=AZURE_DB_NAME)
    container = db.get_container_client('settings')

    # validate temperature, frequency_penalty, presence_penalty
    if temperature < 0 or temperature > 1:
        logging.error(f"[util__module] set_settings: invalid temperature value {temperature}.")
        return

    if frequency_penalty < 0 or frequency_penalty > 1:
        logging.error(f"[util__module] set_settings: invalid frequency_penalty value {frequency_penalty}.")
        return
    
    if presence_penalty < 0 or presence_penalty > 1:
        logging.error(f"[util__module] set_settings: invalid presence_penalty value {presence_penalty}.")
        return
    
    # set default values
    if not temperature:
        temperature = 0.0
    if not frequency_penalty:
        frequency_penalty = 0.0
    if not presence_penalty:
        presence_penalty = 0.0

    if client_principal['id']:
        query = "SELECT * FROM c WHERE c.user_id = @user_id"
        parameters = [
            {"name": "@user_id", "value": client_principal['id']}
        ]

        logging.info(f"[util__module] set_settings: user_id {client_principal['id']}.")

        results = list(container.query_items(
            query=query,
            parameters=parameters,
            enable_cross_partition_query=True
        ))

        if results:
            logging.info(f"[util__module] set_settings: user_id {client_principal['id']} found, results are {results}.")
            setting = results[0]

            setting["temperature"] = temperature
            setting["frequencyPenalty"] = frequency_penalty
            setting["presencePenalty"] = presence_penalty
            try:
                container.replace_item(item=setting['id'], body=setting)
                logging.info(f"Successfully updated settings document for user {client_principal['id']}")
            except Exception as e:
                logging.error(f"Failed to update settings document for user {client_principal['id']}. Error: {str(e)}")
        else:
            logging.info(f"[util__module] set_settings: user_id {client_principal['id']} not found. creating new document.")
            
            try:
                new_setting["id"] = str(uuid.uuid4())
                new_setting["user_id"] = client_principal['id']
                new_setting["temperature"] = temperature
                new_setting["frequencyPenalty"] = frequency_penalty
                new_setting["presencePenalty"] = presence_penalty
                container.create_item(body=new_setting)
                
                logging.info(f"Successfully created new settings document for user {client_principal['id']}")
            except Exception as e:
                logging.error(f"Failed to create settings document for user {client_principal['id']}. Error: {str(e)}")
    else:
        logging.info(f"[util__module] set_settings: user_id not provided.")


##########################################################
# FEEDBACK
##########################################################
def get_feedback_all(client_principal):
    if not client_principal['id']:
        return { "error": "User ID not found." }

    logging.info("User ID found. Getting feedback for user: " + client_principal['id'])

    feedback = []
    credential = DefaultAzureCredential()
    db_client = CosmosClient(AZURE_DB_URI, credential, consistency_level='Session')
    db = db_client.get_database_client(database=AZURE_DB_NAME)
    container = db.get_container_client('feedback')
    
    try:
        query = "SELECT * FROM c WHERE c.user_id = @user_id"
        parameters = [
            {"name": "@user_id", "value": client_principal['id']}
        ]
        result = list(container.query_items(
            query=query,
            parameters=parameters,
            enable_cross_partition_query=True
        ))
        if result:
            feedback = result
    except Exception as e:
        logging.info(f"[util__module] get_feedback_all: something went wrong. {str(e)}")
    return feedback


def get_feedback(conversation_id, client_principal):
    if not client_principal['id'] or not conversation_id:
        return { "error": "User ID or Conversation ID not found." }
    
    logging.info("User ID and Conversation ID found. Getting feedback for user: " + client_principal['id'] + " and conversation: " + conversation_id)

    feedback = {}
    credential = DefaultAzureCredential()
    db_client = CosmosClient(AZURE_DB_URI, credential, consistency_level='Session')
    db = db_client.get_database_client(database=AZURE_DB_NAME)
    container = db.get_container_client('feedback')
    try:
        query = "SELECT * FROM c WHERE c.user_id = @user_id AND c.conversation_id = @conversation_id"
        parameters = [
            {"name": "@user_id", "value": client_principal['id']},
            {"name": "@conversation_id", "value": conversation_id}
        ]
        result = list(container.query_items(
            query=query,
            parameters=parameters,
            enable_cross_partition_query=True
        ))
        if result:
            feedback = result[0]
    except Exception as e:
        logging.info(f"[util__module] get_feedback: something went wrong. {str(e)}")
    return feedback


def set_feedback(client_principal, conversation_id, feedback_message, question, answer, rating, category):
    if not client_principal['id']:
        return { "error": "User ID not found." }
    
    if not conversation_id:
        return { "error": "Conversation ID not found." }

    if not question:
        return { "error": "Question not found." }

    if not answer:
        return { "error": "Answer not found." }

    if rating and rating not in [0, 1]:
        return { "error": "Invalid rating value." }
    
    if feedback_message and len(feedback_message) > 500:
        return { "error": "Feedback message is too long." }

    logging.info("User ID and Conversation ID found. Setting feedback for user: " + client_principal['id'] + " and conversation: " + str(conversation_id))

    feedback = {}
    credential = DefaultAzureCredential()
    db_client = CosmosClient(AZURE_DB_URI, credential, consistency_level='Session')
    db = db_client.get_database_client(database=AZURE_DB_NAME)
    container = db.get_container_client('feedback')
    try:
        feedback = {
            "id": str(uuid.uuid4()),
            "user_id": client_principal['id'],
            "conversation_id": conversation_id,
            "feedback_message": feedback_message,
            "question": question,
            "answer": answer,
            "rating": rating,
            "category": category
        }
        result = container.create_item(body=feedback)
        print("Feedback created: ", result)
    except Exception as e:
        logging.info(f"[util__module] set_feedback: something went wrong. {str(e)}")
    return feedback