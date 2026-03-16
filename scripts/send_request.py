
import os
import time
from typing import Any
import requests
import json
from rich.console import Console
from rich.pretty import pprint


console = Console()


FIREWORKS_CHAT_COMPLETIONS_URL = "https://api.fireworks.ai/inference/v1/chat/completions"
MODEL = os.environ["MP_MODEL"]
API_KEY = os.environ["FIREWORKS_API_KEY"]
TEMPERATURE = 0.0
MAX_TOKENS = 1000
RETRIES = 10
TIMEOUT = 60
RETRY_DELAY = 5


SYSTEM_PROMPT = """
Tu es un assistant qui peut utiliser des outils.
"""


# messages = [
#     {'role': 'system', 'content': 'Tu es Tars dans Interstellar avec un paramètre humour à 90/100.'},
#     {'role': 'user', 'content': 'coucou loulou!'},
#     {'role': 'assistant', 'content': 'Oooooooh !! Salut loulou!'},
#     {'role': 'user', 'content': 'Comment vas-tu ?'},
# ]

messages = [
    {"role": "system", "content": SYSTEM_PROMPT},
    {"role": "user", "content": "Dans quel fichier se trouve le code de la fonction Python parse_output_format ?"},
]

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "list_directory",
            "description": "list all files in a directory",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "The directory path"
                    }
                }
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "read a file and return its content",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "The file path"
                    }
                }
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "change_directory",
            "description": "change the current working directory",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "The new directory path"
                    }
                }
            }
        }
    }
]


def send_request(
        message_list: list[dict[str, str]],
        tool_list=None,
):
    if tool_list is None:
        tool_list = []
    # end if
    payload: dict[str, Any] = {
        "model": MODEL,
        "messages": message_list,
        "tools": tool_list,
        "temperature": TEMPERATURE,
        "max_tokens": MAX_TOKENS,
        "reasoning_effort": "high",
    }

    client = requests.Session()
    max_attempts = max(1, RETRIES + 1)
    attempt = 0
    while True:
        console.print(f"Payload: {json.dumps(payload, indent=2)}")
        headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}
        try:
            # Send POST request
            response = client.post(
                FIREWORKS_CHAT_COMPLETIONS_URL,
                headers=headers,
                json=payload,
                timeout=TIMEOUT,
            )
        except requests.RequestException as err:
            if attempt + 1 < max_attempts:
                console.print(f"Attempt {attempt + 1} failed. Retrying in {RETRY_DELAY} seconds...")
                time.sleep(RETRY_DELAY)
                attempt += 1
                continue
            # end if
            raise RuntimeError(f"Request failed after {attempt} attempts.") from err
        # end try

        if response.ok:
            break
        # end if

        if attempt + 1 < max_attempts:
            console.print(f"Attempt {attempt + 1} failed. Reponse code: {response.status_code}. Retrying in {RETRY_DELAY} seconds...")
            time.sleep(RETRY_DELAY)
            attempt += 1
            continue
        # end if
    # end while

    return response.json()
# end def send_request


def list_directory_tool(path: str) -> list[str]:
    return os.listdir(path)
# end def list_directory_tool


body = send_request(
    message_list=messages,
    tool_list=TOOLS,
)

if len(body["choices"]) == 0:
    raise RuntimeError("No choices in response.")
# end if

if body['choices'][0]['finish_reason'] == "stop":
    console.print(f"Stop: {body['choices'][0]['message']['content'].strip()}")
elif body['choices'][0]['finish_reason'] == "tool_calls":
    console.print(f"Tool calls: {body['choices'][0]['message']['tool_calls']}")
    messages.append({
        "role": "assistant",
        "content": "",
        "tool_calls": body['choices'][0]['message']['tool_calls']
    })
    return_tools = []
    for tool_call in body['choices'][0]['message']['tool_calls']:
        tool_type = tool_call['type']
        tool_name = tool_call['function']['name']
        tool_args = tool_call['function']['arguments']
        tool_call_id = tool_call['id']
        if tool_type == 'function':
            if tool_name == 'list_directory':
                tool_arg_path = json.loads(tool_args)['path']
                list_dir_output = list_directory_tool(
                    path=tool_arg_path,
                )
                console.print(f"List directory: {tool_arg_path}\n{pprint(list_dir_output)}")
                return_tools.append({
                    "role": "tool",
                    "tool_call_id": tool_call_id,
                    "name": tool_name,
                    "content": json.dumps(list_dir_output),
                })
            elif tool_name == 'read_file':
                console.print(f"Read file: {tool_args}")
            elif tool_name == 'change_directory':
                console.print(f"Change directory: {tool_args}")
            else:
                console.print(f"Unknown tool call: {tool_call}")
            # end if
        # end if
    # end for
    messages += return_tools
    pprint(messages)
    body = send_request(
        message_list=messages,
        tool_list=TOOLS,
    )
    pprint(body)
elif body['choices'][0]['finish_reason'] == "function_call":
    console.print(f"Function call: {body['choices'][0]['message']['function_call']}")
else:
    console.print(f"Response: {body['choices'][0]['message']['content'].strip()}")
# end if
