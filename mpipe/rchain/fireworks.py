# SPDX-License-Identifier: GPL-3.0-or-later
#
# mpipe - Multi-provider LLM CLI tools
# Copyright (C) 2026 Nicolas Schaetti and contributors
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.

"""mpipe.rchain.fireworks module.

Notes
-----
This module is part of the Python port of the `mpipe` project.
"""
from __future__ import annotations

import os
from typing import Any, Dict, List

import requests
from rich.pretty import pprint
from rich.console import Console

from mpipe.logging import LogConfig, LogLevels
from mpipe.rchain.chat_runtime import (
    RequestFailureApi,
    RequestFailureRequest,
    RetryConfig,
    send_chat_request_with_retry,
)
from mpipe.rchain.provider import (
    ApiError,
    AskOptions,
    ChatResponse,
    ChatMessage,
    EmptyResponseError,
    MissingApiKeyError,
    Provider,
    RequestError,
    Usage,
    api_key_env, ResponseChoice,
)


console = Console()


FIREWORKS_CHAT_COMPLETIONS_URL = "https://api.fireworks.ai/inference/v1/chat/completions"


async def ask(prompt: str, model: str) -> str:
    """Ask.

    Parameters
    ----------
    prompt : str
        Argument value.
    model : str
        Argument value.

    Returns
    -------
    str
        Returned value.
    """
    response = await ask_messages([ChatMessage.user(prompt)], model, AskOptions())
    return response.content
# end def ask


async def ask_messages(
        messages: list[ChatMessage],
        model: str,
        options: AskOptions,
        log_config: LogConfig | None = None,
) -> ChatResponse:
    """
    Ask messages.

    Parameters
    ----------
    messages : list[ChatMessage]
        Argument value.
    model : str
        Argument value.
    options : AskOptions
        Argument value.
    log_config : LogConfig
        Argument value.

    Returns
    -------
    ChatResponse
        Returned value.
    """
    provider = Provider.FIREWORKS
    key_env = api_key_env(provider)
    api_key = os.getenv(key_env)
    if not api_key:
        raise MissingApiKeyError(provider, key_env)
    # end if

    payload: dict[str, Any] = {
        "model": model,
        "messages": [message.to_json() for message in messages],
    }
    if options.temperature is not None:
        payload["temperature"] = options.temperature
    # end if
    if options.max_tokens is not None:
        payload["max_tokens"] = options.max_tokens
    # end if

    client = requests.Session()
    try:
        response = await send_chat_request_with_retry(
            client,
            FIREWORKS_CHAT_COMPLETIONS_URL,
            api_key,
            payload,
            RetryConfig(
                timeout_secs=options.timeout_secs,
                retries=options.retries,
                retry_delay_ms=options.retry_delay_ms,
            ),
        )
    except RequestFailureRequest as err:
        raise RequestError(provider, err.source) from err
    except RequestFailureApi as err:
        raise ApiError(provider, err.status_code, err.body) from err
    # end try

    try:
        body = response.json()
    except Exception as err:
        raise RequestError(provider, err) from err
    # end try

    level = log_config.level if log_config is not None else LogLevels.NORMAL
    if level >= LogLevels.VERBOSE:
        console.print(f"[bold green]Fireworks API response:[/bold green]")
        pprint(body)
    # end if

    # Extract response information
    res_id, obj, created, model = _extract_response_info(body)

    # Check we have choices
    if len(body.get("choices", [])) == 0:
        raise EmptyResponseError(provider)
    # end if

    # Extract content from the response
    contents: List[ResponseChoice] = []
    for choice in body.get("choices", []):
        content = _extract_message(choice)
        contents.append(content)
        if content:
            break
        # end if
    # end for

    usage_payload = body.get("usage") if isinstance(body, dict) else None
    usage = (
        Usage(
            prompt_tokens=_opt_int(usage_payload, "prompt_tokens"),
            completion_tokens=_opt_int(usage_payload, "completion_tokens"),
            total_tokens=_opt_int(usage_payload, "total_tokens"),
        )
        if isinstance(usage_payload, dict)
        else None
    )
    return ChatResponse(
        response_id=res_id,
        object=obj,
        created=int(created),
        model=model,
        choices=contents,
        usage=usage
    )
# end def ask_messages


def _extract_response_info(body: Any) -> tuple[str, str, str, str]:
    """Extract response info."""
    return (
        body.get("id"),
        body.get("object"),
        body.get("created"),
        body.get("model")
    )
# end def extract_response_info


def _extract_message(choice: Dict[str, Any]) -> ResponseChoice:
    """ extract message."""
    if not isinstance(choice, dict):
        raise ValueError("Invalid choice")
    # end if

    if "message" not in choice:
        raise ValueError("Invalid choice")
    # end if

    response_message = ChatMessage(
        role=choice.get("message").get("role"),
        content=choice.get("message").get("content"),
        reasoning_content=choice.get("message").get("reasoning_content", None)
    )

    return ResponseChoice(
        index=choice.get("index"),
        message=response_message,
        finish_reason=choice.get("finish_reason"),
    )
# end def _extract_message


def _extract_content(body: Any, choice_ix: int) -> str:
    """ extract content.

    Parameters
    ----------
    body : Any
        Argument value.

    Returns
    -------
    str
        Returned value.
    """
    if not isinstance(body, dict):
        return ""
    # end if
    choices = body.get("choices")
    if not isinstance(choices, list) or not choices:
        return ""
    # end if
    first = choices[choice_ix]
    if not isinstance(first, dict):
        return ""
    # end if
    message = first.get("message")
    if not isinstance(message, dict):
        return ""
    # end if
    content = message.get("content")
    return content if isinstance(content, str) else ""
# end def _extract_content


def _opt_int(data: dict[str, Any], key: str) -> int | None:
    """ opt int.

    Parameters
    ----------
    data : dict[str, Any]
        Argument value.
    key : str
        Argument value.

    Returns
    -------
    int | None
        Returned value.
    """
    value = data.get(key)
    return int(value) if isinstance(value, int) else None
# end def _opt_int
