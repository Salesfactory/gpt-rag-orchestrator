import os
import glob
import json
from typing import Any, Dict
from semantic_kernel.orchestration.sk_function_base import SKFunctionBase
from semantic_kernel.utils.validation import validate_skill_name
from semantic_kernel.semantic_functions.prompt_template import PromptTemplate
from semantic_kernel.semantic_functions.prompt_template_config import (
    PromptTemplateConfig,
)
from semantic_kernel.semantic_functions.semantic_function_config import (
    SemanticFunctionConfig,
)

'''
    This function is a modified version of the import_semantic_skill_from_directory function from the semantic_kernel package.
    it will allow us to modify the file from the rag plugin to be able to dynamically load the parameters
    like temperature, frequency_penalty, and presence_penalty from the settings file in the azure function.
'''
def import_semantic_skill_from_directory(
    kernel: Any, parent_directory: str, skill_directory_name: str,
    settings: Dict[str, Any]
) -> Dict[str, SKFunctionBase]:
    CONFIG_FILE = "config.json"
    PROMPT_FILE = "skprompt.txt"

    validate_skill_name(skill_directory_name)

    skill_directory = os.path.join(parent_directory, skill_directory_name)
    skill_directory = os.path.abspath(skill_directory)

    if not os.path.exists(skill_directory):
        raise ValueError(f"Skill directory does not exist: {skill_directory_name}")

    skill = {}

    directories = glob.glob(skill_directory + "/*/")
    for directory in directories:
        dir_name = os.path.dirname(directory)
        function_name = os.path.basename(dir_name)
        prompt_path = os.path.join(directory, PROMPT_FILE)

        # Continue only if the prompt template exists
        if not os.path.exists(prompt_path):
            continue

        config = PromptTemplateConfig()
        config_path = os.path.join(directory, CONFIG_FILE)
        with open(config_path, "r") as config_file:
            # replace settings into config
            temp_config = json.loads(config_file.read())
            temp_config['completion'].update(settings)
            # load config
            config = config.from_json(json.dumps(temp_config))

        # Load Prompt Template
        with open(prompt_path, "r") as prompt_file:
            template = PromptTemplate(
                prompt_file.read(), kernel.prompt_template_engine, config
            )

        # Prepare lambda wrapping AI logic
        function_config = SemanticFunctionConfig(config, template)

        skill[function_name] = kernel.register_semantic_function(
            skill_directory_name, function_name, function_config
        )

    return skill