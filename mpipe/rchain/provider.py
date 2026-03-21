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

"""Shared provider abstractions, request/response models, and errors."""
from __future__ import annotations

import base64
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, List, Dict

from mpipe.logging import LogConfig


class Provider(str, Enum):
    """Supported LLM providers."""
    OPENAI = "openai"
    FIREWORKS = "fireworks"

    def as_str(self) -> str:
        """Return the provider value as a plain string.

        Returns:
            The provider identifier string.
        """
        return self.value
    # end def as_str
# end class Provider


def endpoint(provider: Provider) -> str:
    """Return the chat completions endpoint URL for a provider.

    Args:
        provider: Target provider.

    Returns:
        HTTPS endpoint used for chat completion requests.
    """
    if provider == Provider.OPENAI:
        return "https://api.openai.com/v1/chat/completions"
    # end if
    return "https://api.fireworks.ai/inference/v1/chat/completions"
# end def endpoint


def api_key_env(provider: Provider) -> str:
    """Return the environment variable name containing a provider API key.

    Args:
        provider: Target provider.

    Returns:
        Environment variable name for the provider's API key.
    """
    if provider == Provider.OPENAI:
        return "OPENAI_API_KEY"
    # end if
    return "FIREWORKS_API_KEY"
# end def api_key_env


def is_api_key_present(provider: Provider) -> bool:
    """Check whether the provider API key environment variable is set.

    Args:
        provider: Target provider.

    Returns:
        ``True`` when a non-empty API key is present, else ``False``.
    """
    import os

    value = os.getenv(api_key_env(provider), "")
    return bool(value.strip())
# end def is_api_key_present


@dataclass(slots=True)
class ImageUrl:
    """Simple image URL wrapper used by typed payloads.

    Attributes:
        url: Image URL or data URL string.
    """
    url: str
# end class ImageUrl


class MessageContent:
    """Represents message content as plain text or multimodal parts."""
    def __init__(self, value: str | list[dict[str, Any]]) -> None:
        """Initialize a content container.

        Args:
            value: Plain text content or a list of content parts.
        """
        self.value = value
    # end def __init__

    @classmethod
    def text(cls, text: str) -> "MessageContent":
        """Create text-only message content.

        Args:
            text: Text content.

        Returns:
            A ``MessageContent`` instance containing plain text.
        """
        return cls(text)
    # end def text

    @classmethod
    def with_image(cls, text: str, image_url: str) -> "MessageContent":
        """Create multimodal content with text and one image URL.

        Args:
            text: Text part content.
            image_url: URL or data URL for the image part.

        Returns:
            A ``MessageContent`` instance with two structured parts.
        """
        return cls(
            [
                {"type": "text", "text": text},
                {"type": "image_url", "image_url": {"url": image_url}},
            ]
        )
    # end def with_image

    def is_empty(self) -> bool:
        """Return whether the content is empty.

        Returns:
            ``True`` if text is empty or part list is empty, else ``False``.
        """
        if isinstance(self.value, str):
            return self.value == ""
        # end if
        return len(self.value) == 0
    # end def is_empty

    def text_len(self) -> int:
        """Compute a text length approximation for logging and diagnostics.

        Returns:
            Number of text characters, counting text parts only.
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
        """Return the underlying JSON-compatible content payload.

        Returns:
            Plain text or a list of multimodal content parts.
        """
        return self.value
    # end def to_json
# end class MessageContent


@dataclass(slots=True)
class ChatMessage:
    """Normalized chat message payload.

    Attributes:
        role: Message role (``system``, ``user``, or ``assistant``/others).
        content: Message content object.
        reasoning_content: Optional reasoning payload returned by some models.
    """
    role: str
    content: MessageContent
    reasoning_content: str | None = None

    @classmethod
    def system(cls, content: MessageContent | str) -> "ChatMessage":
        """Create a system-role message.

        Args:
            content: Either plain text or prebuilt message content.

        Returns:
            A ``ChatMessage`` with role ``system``.
        """
        if isinstance(content, str):
            content = MessageContent.text(content)
        # end if
        return cls(role="system", content=content)
    # end def system

    @classmethod
    def user(cls, content: MessageContent | str) -> "ChatMessage":
        """Create a user-role message.

        Args:
            content: Either plain text or prebuilt message content.

        Returns:
            A ``ChatMessage`` with role ``user``.
        """
        if isinstance(content, str):
            content = MessageContent.text(content)
        # end if
        return cls(role="user", content=content)
    # end def user

    @classmethod
    def user_with_text(cls, text: str) -> "ChatMessage":
        """Create a user-role message from text.

        Args:
            text: User text message.

        Returns:
            A ``ChatMessage`` wrapping text content.
        """
        return cls.user(MessageContent.text(text))
    # end def user_with_text

    @classmethod
    def user_with_text_and_image(cls, text: str, image_url: str) -> "ChatMessage":
        """Create a user-role multimodal message.

        Args:
            text: User text.
            image_url: URL or data URL of the image.

        Returns:
            A ``ChatMessage`` with text and image content parts.
        """
        return cls.user(MessageContent.with_image(text, image_url))
    # end def user_with_text_and_image

    def to_json(self) -> dict[str, Any]:
        """Serialize the chat message to a JSON-compatible dictionary.

        Returns:
            Message payload suitable for provider APIs.
        """
        return {"role": self.role, "content": self.content.to_json(), "reasoning_content": self.reasoning_content}
    # end def to_json
# end class ChatMessage


def resolve_image_url(input_value: str) -> str:
    """Resolve an input image reference to a URL or data URL.

    Args:
        input_value: HTTP(S) URL or local file path.

    Returns:
        The original HTTP(S) URL, or an encoded ``data:`` URL for local files.

    Raises:
        ValueError: If a local path does not exist or cannot be read.
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
class StructuredOutputFormat:
    """Structured output configuration sent to a provider.

    Attributes:
        type: Provider-specific response format type.
        name: Logical schema name.
        schema_key: Key used to nest schema details in provider payloads.
        schema: JSON schema dictionary.
    """
    type: str
    name: str
    schema_key: str | None = None
    schema: dict[str, Any] | None = None
# end class ResponseFormat


class StructuredOutputFormatJSON(StructuredOutputFormat):
    """JSON schema response format specialization."""

    type: str = "json_schema"
    schema_key: str = "json_schema"

    def __init__(
            self,
            name: str,
            json_schema: dict[str, Any]
    ) -> None:
        """Initialize JSON schema response format.

        Args:
            name: Logical schema name.
            json_schema: JSON schema dictionary.
        """
        self.name = name
        self.schema = json_schema
    # end def __init__

# end class StructuredOutputFormatJSON


@dataclass(slots=True)
class ChatOptions:
    """Runtime options controlling provider chat requests.

    Attributes:
        temperature: Optional sampling temperature.
        max_tokens: Optional generation token limit.
        timeout_secs: Optional request timeout in seconds.
        retries: Number of retry attempts for transient failures.
        retry_delay_ms: Base delay in milliseconds between retries.
    """
    temperature: float | None = None
    max_tokens: int | None = None
    timeout_secs: int | None = None
    retries: int = 0
    retry_delay_ms: int = 500
# end class AskOptions


@dataclass(slots=True)
class Usage:
    """Token usage metrics returned by providers.

    Attributes:
        prompt_tokens: Prompt token count.
        completion_tokens: Completion token count.
        total_tokens: Total token count.
    """
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None
# end class Usage


@dataclass(slots=True)
class ResponseChoice:
    """One completion candidate returned by a provider.

    Attributes:
        index: Choice index in provider response.
        message: Message payload for this choice.
        finish_reason: Optional stop reason returned by provider.
        token_ids: Optional token IDs returned by some providers.
    """
    index: int
    message: ChatMessage
    finish_reason: str | None = None
    token_ids: list[int] | None = None
# end class ResponseChoice


@dataclass(slots=True)
class ChatResponse:
    """Normalized provider response containing one or more message choices.

    Attributes:
        response_id: Provider response identifier.
        object: Provider response object type.
        created: Creation timestamp.
        model: Model ID used for generation.
        choices: Completion choices returned by the provider.
        usage: Optional token usage information.
    """
    response_id: str
    object: str
    created: int
    model: str
    choices: List[ResponseChoice]
    usage: Usage | None = None

    def get_message(self, index: int = 0) -> ChatMessage | None:
        """Return the message at a specific choice index.

        Args:
            index: Choice index to retrieve.

        Returns:
            The selected message, or ``None`` when no choices exist.
        """
        if len(self.choices) == 0:
            return None
        # end if
        return self.choices[index].message
    # end def get_message

    def n_messages(self) -> int:
        """Return the number of completion choices in this response.

        Returns:
            Number of available choices.
        """
        return len(self.choices)
    # end def n_messages

# end class ChatResponse


class ProviderError(Exception):
    """Base class for provider-related runtime errors."""
    pass
# end class ProviderError


class MissingApiKeyError(ProviderError):
    """Raised when a required provider API key is missing."""
    def __init__(self, provider: Provider, key_env: str) -> None:
        """Initialize a missing API key error.

        Args:
            provider: Provider requiring the key.
            key_env: Environment variable name expected to contain the key.
        """
        super().__init__(f"{key_env} is not set in the environment")
        self.provider = provider
        self.key_env = key_env
    # end def __init__
# end class MissingApiKeyError


class RequestError(ProviderError):
    """Raised when a transport-level request failure occurs."""
    def __init__(self, provider: Provider, source: Exception) -> None:
        """Initialize a request error.

        Args:
            provider: Provider that failed.
            source: Original transport exception.
        """
        super().__init__(f"{provider.as_str()} request failed: {source}")
        self.provider = provider
        self.source = source
    # end def __init__
# end class RequestError


class ApiError(ProviderError):
    """Raised when a provider returns a non-success API response."""
    def __init__(self, provider: Provider, status: int, body: str) -> None:
        """Initialize an API error.

        Args:
            provider: Provider that returned the error.
            status: HTTP status code.
            body: Response body text.
        """
        super().__init__(f"{provider.as_str()} API error {status}: {body}")
        self.provider = provider
        self.status = status
        self.body = body
    # end def __init__
# end class ApiError


class EmptyResponseError(ProviderError):
    """Raised when a provider response contains no usable message content."""
    def __init__(self, provider: Provider) -> None:
        """Initialize an empty response error.

        Args:
            provider: Provider that returned an empty payload.
        """
        super().__init__(f"{provider.as_str()} response did not contain message content")
        self.provider = provider
    # end def __init__
# end class EmptyResponseError


async def ask(
        provider: Provider,
        model: str,
        messages: list[ChatMessage],
        options: ChatOptions,
        structured_output: StructuredOutputFormat | None = None,
        log_config: LogConfig | None = None,
) -> ChatResponse:
    """Dispatch a chat request to the selected provider backend.

    Args:
        provider: Provider to query.
        model: Model identifier.
        messages: Chat history/messages.
        options: Runtime request options.
        structured_output: Optional structured output specification.
        log_config: Optional logging verbosity configuration.

    Returns:
        A normalized ``ChatResponse`` object.
    """
    from mpipe.rchain import fireworks, openai

    if provider == Provider.OPENAI:
        return await openai.ask_messages(
            messages=messages,
            model=model,
            options=options,
            structured_output=structured_output,
            log_config=log_config
        )
    # end if
    return await fireworks.ask_messages(
        messages=messages,
        model=model,
        options=options,
        structured_output=structured_output,
        log_config=log_config
    )
# end def ask
