
""" 
In this module, we will be attempting to use Azure OpenAI 
Authenticated with Microsoft Entra ID
to avoid exposing the API key in the code.
"""
import sys
import os 
from azure.identity import DefaultAzureCredential, get_bearer_token_provider
from openai import AzureOpenAI
from pydantic import BaseModel, Field
from typing import Dict, Union
import logging
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from openai import APIError, RateLimitError, APITimeoutError

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [AzureOpenAI] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)



class LLMConfig(BaseModel):

    """Configuration for the Azure OpenAI API for Microsoft Entra ID Authentication """

    api_base: str = Field(default=os.getenv("AZURE_OPENAI_API_BASE"), description="The base URL for the Azure OpenAI API")
    api_version: str = Field(default=os.getenv("AZURE_OPENAI_API_VERSION"), description="The API version for the Azure OpenAI API")
    deployment_name: str = Field(..., description="The name of the Azure OpenAI model to use")
    api_key: str = Field(default=os.getenv("AZURE_OPENAI_API_KEY"), description="The API key for the Azure OpenAI API in case of no Microsoft Entra ID Authentication")

    class Config: 
        frozen = True # makes the config object immutable 
    
    def get_token(self):
        """
        Get a token for the Azure OpenAI API
        """
        logger.info("Attempting to get bearer token for Azure OpenAI API")
        try:
            token = get_bearer_token_provider(
                DefaultAzureCredential(), "https://cognitiveservices.azure.com/.default")
            logger.info("Successfully obtained bearer token")
            return token
        except Exception as e:
            logger.error(f"Failed to get bearer token: {str(e)}", exc_info=True)
            raise
    
class PromptTemplate(BaseModel): 
    
    """ 
    This class is a collection of system prompts 
    """
    
    basic_system_prompt: str = Field(default = "You're a helpful assistant")

    creative_system_prompt: str = Field(default = "You're a creative assistant, answer questions in a creative way")

    class Config: 
        frozen = True

class LLMManager:
    def __init__(self, deployment_name: str):
        logger.info(f"Initializing LLMManager with deployment: {deployment_name}")
        logger.info("Setting up PromptTemplate and configuration")
        self.prompts = PromptTemplate()
        self._clients: Dict[str, Union[AzureOpenAI]] = {}
        self.config = LLMConfig(deployment_name=deployment_name)
        logger.info("LLMManager initialization complete")
    
    def get_client(self, client_type: str, use_langchain: bool = False) -> Union[AzureOpenAI]:
        logger.info(f"Getting client of type: {client_type} (use_langchain: {use_langchain})")
        
        """
        Get or create an Azure OpenAI client
        
        Args: 
            client_type: Type of client to create (chat model or embedding model)
            use_langchain: whether to use langchain or not
        """
        try:
            logger.info("Creating standard Azure OpenAI client")
            client = AzureOpenAI(
                azure_ad_token_provider=self.config.get_token(), 
                api_version=self.config.api_version,
                azure_endpoint=self.config.api_base,
            )
            logger.info(f"Successfully created client: {type(client).__name__}")
            self._clients[client_type] = client
            return client
        except Exception as e:
            logger.error(f"Failed to create client: {str(e)}", exc_info=True)
            raise
    
    def get_prompt(self, prompt_type: str) -> str:
        """
        Get a system prompt
        """
        logger.info(f"Retrieving prompt of type: {prompt_type}")
        try:
            prompt = getattr(self.prompts, prompt_type)
            logger.info(f"Successfully retrieved prompt: {prompt[:50]}...")
            return prompt
        except AttributeError as e:
            logger.error(f"Unknown prompt type: {prompt_type}")
            raise ValueError(f"Unknown prompt type: {prompt_type}")
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type((APIError, RateLimitError, APITimeoutError)),
        before_sleep=lambda retry_state: logger.warning(f"Retrying after error. Attempt {retry_state.attempt_number}/3")
    )
    def _make_chat_request(self, client, prompt, user_message):
        try:
            response = client.chat.completions.create(
                model=self.config.deployment_name,
                messages=[
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": user_message}
                ],
                max_tokens=600
            )
            return response.choices[0].message.content
        except (APIError, RateLimitError, APITimeoutError) as e:
            logger.error(f"OpenAI API error: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error during chat completion: {str(e)}")
            raise

    def get_response(self, prompt_type: str = None, client_type: str = None, use_langchain: bool = False, custom_prompt: str = None, user_message: str = "Hello!") -> str:
        logger.info(f"Getting response using prompt_type: {prompt_type}, client_type: {client_type}")
        try:
            client = self.get_client(client_type, use_langchain=use_langchain)
            prompt = custom_prompt if custom_prompt else self.get_prompt(prompt_type)
            logger.info("Using standard Azure OpenAI client for request")
            return self._make_chat_request(client, prompt, user_message)
                
        except Exception as e:
            logger.error(f"Error getting response: {str(e)}", exc_info=True)
            raise

if __name__ == "__main__":
    # llm_manager = LLMManager(deployment_name="Agent")
    # print(llm_manager.get_response("basic_system_prompt", "Agent"))
    # test token
    llm_manager = LLMManager(deployment_name="Agent")
    print(llm_manager.get_response(user_message="Hello!", prompt_type="basic_system_prompt", client_type="Agent"))
