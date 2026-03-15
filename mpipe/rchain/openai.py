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

"""mpipe.rchain.openai module.

Notes
-----
This module is part of the Python port of the `mpipe` project.
"""
from __future__ import annotations

import os
from typing import Any

import requests

from mpipe.rchain.chat_runtime import (
    RequestFailureApi,
    RequestFailureRequest,
    RetryConfig,
    send_chat_request_with_retry,
)
from mpipe.rchain.provider import (
    ApiError,
    AskOptions,
    AskResponse,
    ChatMessage,
    EmptyResponseError,
    MissingApiKeyError,
    Provider,
    RequestError,
    Usage,
    api_key_env,
)


OPENAI_CHAT_COMPLETIONS_URL = "https://api.openai.com/v1/chat/completions"


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
) -> AskResponse:
    """Ask messages.

    Parameters
    ----------
    messages : list[ChatMessage]
        Argument value.
    model : str
        Argument value.
    options : AskOptions
        Argument value.

    Returns
    -------
    AskResponse
        Returned value.
    """
    provider = Provider.OPENAI
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
            OPENAI_CHAT_COMPLETIONS_URL,
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

    content = _extract_content(body)
    if not content:
        raise EmptyResponseError(provider)
    # end if

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
    return AskResponse(content=content, usage=usage)
# end def ask_messages


def _extract_content(body: Any) -> str:
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
    first = choices[0]
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
