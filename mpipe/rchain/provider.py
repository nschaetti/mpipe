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

"""mpipe.rchain.provider module.

Notes
-----
This module is part of the Python port of the `mpipe` project.
"""
from __future__ import annotations

import base64
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any


class Provider(str, Enum):
    """Provider.

    Notes
    -----
    This class follows the same role and hierarchy as the Rust implementation.
    """
    OPENAI = "openai"
    FIREWORKS = "fireworks"

    def as_str(self) -> str:
        """As str.

        Parameters
        ----------
        None
            This callable does not accept explicit parameters.

        Returns
        -------
        str
            Returned value.
        """
        return self.value
    # end def as_str
# end class Provider


def endpoint(provider: Provider) -> str:
    """Endpoint.

    Parameters
    ----------
    provider : Provider
        Argument value.

    Returns
    -------
    str
        Returned value.
    """
    if provider == Provider.OPENAI:
        return "https://api.openai.com/v1/chat/completions"
    # end if
    return "https://api.fireworks.ai/inference/v1/chat/completions"
# end def endpoint


def api_key_env(provider: Provider) -> str:
    """Api key env.

    Parameters
    ----------
    provider : Provider
        Argument value.

    Returns
    -------
    str
        Returned value.
    """
    if provider == Provider.OPENAI:
        return "OPENAI_API_KEY"
    # end if
    return "FIREWORKS_API_KEY"
# end def api_key_env


def is_api_key_present(provider: Provider) -> bool:
    """Is api key present.

    Parameters
    ----------
    provider : Provider
        Argument value.

    Returns
    -------
    bool
        Returned value.
    """
    import os

    value = os.getenv(api_key_env(provider), "")
    return bool(value.strip())
# end def is_api_key_present


@dataclass(slots=True)
class ImageUrl:
    """Imageurl.

    Notes
    -----
    This class follows the same role and hierarchy as the Rust implementation.
    """
    url: str
# end class ImageUrl


class MessageContent:
    """Messagecontent.

    Notes
    -----
    This class follows the same role and hierarchy as the Rust implementation.
    """
    def __init__(self, value: str | list[dict[str, Any]]) -> None:
        """__init__.

        Parameters
        ----------
        value : str | list[dict[str, Any]]
            Argument value.

        Returns
        -------
        None
            Returned value.
        """
        self.value = value
    # end def __init__

    @classmethod
    def text(cls, text: str) -> "MessageContent":
        """Text.

        Parameters
        ----------
        text : str
            Argument value.

        Returns
        -------
        'MessageContent'
            Returned value.
        """
        return cls(text)
    # end def text

    @classmethod
    def with_image(cls, text: str, image_url: str) -> "MessageContent":
        """With image.

        Parameters
        ----------
        text : str
            Argument value.
        image_url : str
            Argument value.

        Returns
        -------
        'MessageContent'
            Returned value.
        """
        return cls(
            [
                {"type": "text", "text": text},
                {"type": "image_url", "image_url": {"url": image_url}},
            ]
        )
    # end def with_image

    def is_empty(self) -> bool:
        """Is empty.

        Parameters
        ----------
        None
            This callable does not accept explicit parameters.

        Returns
        -------
        bool
            Returned value.
        """
        if isinstance(self.value, str):
            return self.value == ""
        # end if
        return len(self.value) == 0
    # end def is_empty

    def text_len(self) -> int:
        """Text len.

        Parameters
        ----------
        None
            This callable does not accept explicit parameters.

        Returns
        -------
        int
            Returned value.
        """
        if isinstance(self.value, str):
            return len(self.value)
        # end if
        total = 0
        for part in self.value:
            if part.get("type") == "text":
                total += len(str(part.get("text", "")))
            # end if
        # end for
        return total
    # end def text_len

    def to_json(self) -> str | list[dict[str, Any]]:
        """To json.

        Parameters
        ----------
        None
            This callable does not accept explicit parameters.

        Returns
        -------
        str | list[dict[str, Any]]
            Returned value.
        """
        return self.value
    # end def to_json
# end class MessageContent


@dataclass(slots=True)
class ChatMessage:
    """
    Chatmessage.

    Notes
    -----
    This class follows the same role and hierarchy as the Rust implementation.
    """
    role: str
    content: MessageContent

    @classmethod
    def system(cls, content: MessageContent | str) -> "ChatMessage":
        """System.

        Parameters
        ----------
        content : MessageContent | str
            Argument value.

        Returns
        -------
        'ChatMessage'
            Returned value.
        """
        if isinstance(content, str):
            content = MessageContent.text(content)
        # end if
        return cls(role="system", content=content)
    # end def system

    @classmethod
    def user(cls, content: MessageContent | str) -> "ChatMessage":
        """User.

        Parameters
        ----------
        content : MessageContent | str
            Argument value.

        Returns
        -------
        'ChatMessage'
            Returned value.
        """
        if isinstance(content, str):
            content = MessageContent.text(content)
        # end if
        return cls(role="user", content=content)
    # end def user

    @classmethod
    def user_with_text(cls, text: str) -> "ChatMessage":
        """User with text.

        Parameters
        ----------
        text : str
            Argument value.

        Returns
        -------
        'ChatMessage'
            Returned value.
        """
        return cls.user(MessageContent.text(text))
    # end def user_with_text

    @classmethod
    def user_with_text_and_image(cls, text: str, image_url: str) -> "ChatMessage":
        """User with text and image.

        Parameters
        ----------
        text : str
            Argument value.
        image_url : str
            Argument value.

        Returns
        -------
        'ChatMessage'
            Returned value.
        """
        return cls.user(MessageContent.with_image(text, image_url))
    # end def user_with_text_and_image

    def to_json(self) -> dict[str, Any]:
        """To json.

        Parameters
        ----------
        None
            This callable does not accept explicit parameters.

        Returns
        -------
        dict[str, Any]
            Returned value.
        """
        return {"role": self.role, "content": self.content.to_json()}
    # end def to_json
# end class ChatMessage


def resolve_image_url(input_value: str) -> str:
    """Resolve image url.

    Parameters
    ----------
    input_value : str
        Argument value.

    Returns
    -------
    str
        Returned value.
    """
    trimmed = input_value.strip()
    if trimmed.startswith("http://") or trimmed.startswith("https://"):
        return trimmed
    # end if

    path = Path(trimmed)
    if not path.exists():
        raise ValueError(f"Image file not found: {trimmed}")
    # end if

    ext = path.suffix.lower()
    mime = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".gif": "image/gif",
        ".webp": "image/webp",
        ".bmp": "image/bmp",
    }.get(ext, "application/octet-stream")
    try:
        data = path.read_bytes()
    except OSError as err:
        raise ValueError(f"Failed to read image file: {err}") from err
    # end try
    encoded = base64.b64encode(data).decode("ascii")
    return f"data:{mime};base64,{encoded}"
# end def resolve_image_url


@dataclass(slots=True)
class AskOptions:
    """Askoptions.

    Notes
    -----
    This class follows the same role and hierarchy as the Rust implementation.
    """
    temperature: float | None = None
    max_tokens: int | None = None
    timeout_secs: int | None = None
    retries: int = 0
    retry_delay_ms: int = 500
# end class AskOptions


@dataclass(slots=True)
class Usage:
    """Usage.

    Notes
    -----
    This class follows the same role and hierarchy as the Rust implementation.
    """
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None
# end class Usage


@dataclass(slots=True)
class AskResponse:
    """Askresponse.

    Notes
    -----
    This class follows the same role and hierarchy as the Rust implementation.
    """
    content: str
    usage: Usage | None = None
# end class AskResponse


class ProviderError(Exception):
    """Providererror.

    Notes
    -----
    This class follows the same role and hierarchy as the Rust implementation.
    """
    pass
# end class ProviderError


class MissingApiKeyError(ProviderError):
    """Missingapikeyerror.

    Notes
    -----
    This class follows the same role and hierarchy as the Rust implementation.
    """
    def __init__(self, provider: Provider, key_env: str) -> None:
        """__init__.

        Parameters
        ----------
        provider : Provider
            Argument value.
        key_env : str
            Argument value.

        Returns
        -------
        None
            Returned value.
        """
        super().__init__(f"{key_env} is not set in the environment")
        self.provider = provider
        self.key_env = key_env
    # end def __init__
# end class MissingApiKeyError


class RequestError(ProviderError):
    """Requesterror.

    Notes
    -----
    This class follows the same role and hierarchy as the Rust implementation.
    """
    def __init__(self, provider: Provider, source: Exception) -> None:
        """__init__.

        Parameters
        ----------
        provider : Provider
            Argument value.
        source : Exception
            Argument value.

        Returns
        -------
        None
            Returned value.
        """
        super().__init__(f"{provider.as_str()} request failed: {source}")
        self.provider = provider
        self.source = source
    # end def __init__
# end class RequestError


class ApiError(ProviderError):
    """Apierror.

    Notes
    -----
    This class follows the same role and hierarchy as the Rust implementation.
    """
    def __init__(self, provider: Provider, status: int, body: str) -> None:
        """__init__.

        Parameters
        ----------
        provider : Provider
            Argument value.
        status : int
            Argument value.
        body : str
            Argument value.

        Returns
        -------
        None
            Returned value.
        """
        super().__init__(f"{provider.as_str()} API error {status}: {body}")
        self.provider = provider
        self.status = status
        self.body = body
    # end def __init__
# end class ApiError


class EmptyResponseError(ProviderError):
    """Emptyresponseerror.

    Notes
    -----
    This class follows the same role and hierarchy as the Rust implementation.
    """
    def __init__(self, provider: Provider) -> None:
        """__init__.

        Parameters
        ----------
        provider : Provider
            Argument value.

        Returns
        -------
        None
            Returned value.
        """
        super().__init__(f"{provider.as_str()} response did not contain message content")
        self.provider = provider
    # end def __init__
# end class EmptyResponseError


async def ask(
    provider: Provider,
    model: str,
    messages: list[ChatMessage],
    options: AskOptions,
) -> AskResponse:
    """Ask.

    Parameters
    ----------
    provider : Provider
        Argument value.
    model : str
        Argument value.
    messages : list[ChatMessage]
        Argument value.
    options : AskOptions
        Argument value.

    Returns
    -------
    AskResponse
        Returned value.
    """
    from mpipe.rchain import fireworks, openai

    if provider == Provider.OPENAI:
        return await openai.ask_messages(messages, model, options)
    # end if
    return await fireworks.ask_messages(messages, model, options)
# end def ask
