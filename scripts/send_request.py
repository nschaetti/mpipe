
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
        messages: list[dict[str, str]],
        tools: list[dict[str, Any]] = [],
):
    payload: dict[str, Any] = {
        "model": MODEL,
        "messages": messages,
        "tools": tools,
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

body = send_request(messages)

if len(body["choices"]) == 0:
    raise RuntimeError("No choices in response.")
# end if

if body['choices'][0]['finish_reason'] == "stop":
    console.print(f"Stop: {body['choices'][0]['message']['content'].strip()}")
elif body['choices'][0]['finish_reason'] == "tool_calls":
    console.print(f"Tool calls: {body['choices'][0]['message']['tool_calls']}")
    for tool_call in body['choices'][0]['message']['tool_calls']:
        if tool_call['type'] == 'function':
            if tool_call['function']['name'] == 'list_directory':
                list_dir_output = os.listdir(tool_call['function']['arguments']['path'])

            elif tool_call['function']['name'] == 'read_file':
                console.print(f"Read file: {tool_call['function']['arguments']}")
            elif tool_call['function']['name'] == 'change_directory':
                console.print(f"Change directory: {tool_call['function']['arguments']}")
            else:
                console.print(f"Unknown tool call: {tool_call}")
            # end if
        # end if
    # end for
elif body['choices'][0]['finish_reason'] == "function_call":
    console.print(f"Function call: {body['choices'][0]['message']['function_call']}")
else:
    console.print(f"Response: {body['choices'][0]['message']['content'].strip()}")
# end if
