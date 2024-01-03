from typing import Dict, List
import json
from inspect import Parameter
from pydantic import create_model
import openai

class Apputils:

    @staticmethod
    def generate_json_schema(f) -> Dict:
        """
        Generate a JSON schema for the input parameters of the given function.

        Parameters:
            f (callable): The function for which to generate the JSON schema.

        Returns:
            Dict: A dictionary containing the function name, description, and parameters schema.
        """
        parameters = inspect.signature(f).parameters
        parameters_schema = {}

        for param_name, param_obj in parameters.items():
            param_type = param_obj.annotation
            param_default = param_obj.default

            param_info = {
                "type": param_type.__name__,
                "default": param_default if param_default != Parameter.empty else None,
            }

            parameters_schema[param_name] = param_info

        input_model = create_model(f'Input for `{f.__name__}`', **parameters_schema)
        schema = input_model.schema()

        return {
            "name": f.__name__,
            "description": f.__doc__,
            "parameters": schema,
        }

    @staticmethod
    def wrap_functions(functions: List[callable]) -> List[Dict]:
        """
        Wrap a list of functions and generate JSON schemas for each.

        Parameters:
            functions (List[callable]): List of functions for which to generate JSON schemas.

        Returns:
            List[Dict]: A list of dictionaries, each containing the function name, description, and parameters schema.
        """
        return [Apputils.generate_json_schema(f) for f in functions]

    @staticmethod
    def execute_json_function(response) -> List:
        """
        Execute a function based on the response from an OpenAI ChatCompletion API call.

        Parameters:
            response: The response object from the OpenAI ChatCompletion API call.

        Returns:
            List: The result of the executed function.
        """
        func_name: str = response.choices[0].message.function_call.name
        func_args: Dict = json.loads(
            response.choices[0].message.function_call.arguments)
        function_map = {
            'retrieve_web_search_results': WebSearch.retrieve_web_search_results,
            'web_search_text': WebSearch.web_search_text,
            'web_search_pdf': WebSearch.web_search_pdf,
            'web_search_image': WebSearch.web_search_image,
            'web_search_video': WebSearch.web_search_video,
            'web_search_news': WebSearch.web_search_news,
            'get_instant_web_answer': WebSearch.get_instant_web_answer,
            'web_search_map': WebSearch.web_search_map,
        }
        
        if func_name in function_map:
            result = function_map[func_name](**func_args)
        else:
            raise ValueError(f"Function '{func_name}' not found.")
        return result

    @staticmethod
    def ask_llm_function_caller(gpt_model: str, temperature: float, messages: List, function_json_list: List[Dict]):
        """
        Generate a response from an OpenAI ChatCompletion API call with specific function calls.

        Parameters:
            - gpt_model (str): The name of the GPT model to use.
            - temperature (float): The temperature parameter for the API call.
            - messages (List): List of message objects for the conversation.
            - function_json_list (List[Dict]): List of function JSON schemas.

        Returns:
            The response object from the OpenAI ChatCompletion API call.
        """
        response = openai.ChatCompletion.create(
            engine=gpt_model,
            messages=messages,
            functions=function_json_list,
            function_call="auto",
            temperature=temperature
        )
        return response

    @staticmethod
    def ask_llm_chatbot(gpt_model: str, temperature: float, messages: List):
        """
        Generate a response from an OpenAI ChatCompletion API call without specific function calls.

        Parameters:
            - gpt_model (str): The name of the GPT model to use.
            - temperature (float): The temperature parameter for the API call.
            - messages (List): List of message objects for the conversation.

        Returns:
            The response object from the OpenAI ChatCompletion API call.
        """
        response = openai.ChatCompletion.create(
            engine=gpt_model,
            messages=messages,
            temperature=temperature
        )
        return response
