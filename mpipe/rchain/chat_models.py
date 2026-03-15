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

"""mpipe.rchain.chat_models module.

Notes
-----
This module is part of the Python port of the `mpipe` project.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from enum import Enum
from typing import Any

import requests

from mpipe.rchain.ai import AIMessage
from mpipe.rchain.human import HumanMessage
from mpipe.rchain.tools import ToolCall, ToolDefinition


class MessageRole(str, Enum):
    """Messagerole.

    Notes
    -----
    This class follows the same role and hierarchy as the Rust implementation.
    """
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"
# end class MessageRole


@dataclass(slots=True)
class ChatMessage:
    """Chatmessage.

    Notes
    -----
    This class follows the same role and hierarchy as the Rust implementation.
    """
    role: MessageRole
    content: Any
    tool_call_id: str | None = None
    tool_calls: list[ToolCall] | None = None

    @classmethod
    def user(cls, message: HumanMessage) -> "ChatMessage":
        """User.

        Parameters
        ----------
        message : HumanMessage
            Argument value.

        Returns
        -------
        'ChatMessage'
            Returned value.
        """
        return cls(role=MessageRole.USER, content=message.to_json())
    # end def user

    @classmethod
    def user_text(cls, content: str) -> "ChatMessage":
        """User text.

        Parameters
        ----------
        content : str
            Argument value.

        Returns
        -------
        'ChatMessage'
            Returned value.
        """
        return cls(role=MessageRole.USER, content=content)
    # end def user_text

    @classmethod
    def user_parts(cls, parts: list[dict[str, Any]]) -> "ChatMessage":
        """User parts.

        Parameters
        ----------
        parts : list[dict[str, Any]]
            Argument value.

        Returns
        -------
        'ChatMessage'
            Returned value.
        """
        return cls(role=MessageRole.USER, content=parts)
    # end def user_parts

    @classmethod
    def assistant_from_ai(cls, message: AIMessage) -> "ChatMessage":
        """Assistant from ai.

        Parameters
        ----------
        message : AIMessage
            Argument value.

        Returns
        -------
        'ChatMessage'
            Returned value.
        """
        content = None if message.content == "" else message.content
        calls = message.tool_calls if message.tool_calls else None
        return cls(role=MessageRole.ASSISTANT, content=content, tool_calls=calls)
    # end def assistant_from_ai

    @classmethod
    def tool_result(cls, tool_call_id: str, content: str) -> "ChatMessage":
        """Tool result.

        Parameters
        ----------
        tool_call_id : str
            Argument value.
        content : str
            Argument value.

        Returns
        -------
        'ChatMessage'
            Returned value.
        """
        return cls(role=MessageRole.TOOL, content=content, tool_call_id=tool_call_id)
    # end def tool_result

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
        payload: dict[str, Any] = {"role": self.role.value, "content": self.content}
        if self.tool_call_id:
            payload["tool_call_id"] = self.tool_call_id
        # end if
        if self.tool_calls:
            payload["tool_calls"] = [call.to_json() for call in self.tool_calls]
        # end if
        return payload
    # end def to_json
# end class ChatMessage


@dataclass(slots=True)
class ChatFireworks:
    """Chatfireworks.

    Notes
    -----
    This class follows the same role and hierarchy as the Rust implementation.
    """
    model: str
    temperature: float
    api_key: str
    base_url: str = "https://api.fireworks.ai/inference/v1/chat/completions"
    tools: list[ToolDefinition] | None = None

    @classmethod
    def new(cls, model: str, temperature: float) -> "ChatFireworks":
        """New.

        Parameters
        ----------
        model : str
            Argument value.
        temperature : float
            Argument value.

        Returns
        -------
        'ChatFireworks'
            Returned value.
        """
        api_key = os.getenv("FIREWORKS_API_KEY", "")
        if not api_key:
            raise ValueError("FIREWORKS_API_KEY is not set in the environment")
        # end if
        return cls(model=model, temperature=temperature, api_key=api_key)
    # end def new

    def bind_tools(self, tools: list[ToolDefinition]) -> "ChatFireworks":
        """Bind tools.

        Parameters
        ----------
        tools : list[ToolDefinition]
            Argument value.

        Returns
        -------
        'ChatFireworks'
            Returned value.
        """
        return ChatFireworks(
            model=self.model,
            temperature=self.temperature,
            api_key=self.api_key,
            base_url=self.base_url,
            tools=list(tools),
        )
    # end def bind_tools

    def invoke(self, messages: list[HumanMessage]) -> AIMessage:
        """Invoke.

        Parameters
        ----------
        messages : list[HumanMessage]
            Argument value.

        Returns
        -------
        AIMessage
            Returned value.
        """
        chat_messages = [ChatMessage.user(message) for message in messages]
        return self.invoke_messages(chat_messages)
    # end def invoke

    def invoke_messages(self, messages: list[ChatMessage]) -> AIMessage:
        """Invoke messages.

        Parameters
        ----------
        messages : list[ChatMessage]
            Argument value.

        Returns
        -------
        AIMessage
            Returned value.
        """
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [message.to_json() for message in messages],
            "temperature": self.temperature,
        }
        if self.tools is not None:
            payload["tools"] = [tool.to_json() for tool in self.tools]
        # end if

        response = requests.post(
            self.base_url,
            headers={"Authorization": f"Bearer {self.api_key}"},
            json=payload,
            timeout=60,
        )
        if not response.ok:
            raise RuntimeError(f"Fireworks API error {response.status_code}: {response.text}")
        # end if

        body = response.json()
        message = body.get("choices", [{}])[0].get("message", {})
        content = message.get("content") or ""
        tool_calls = _parse_tool_calls(message)
        return AIMessage(content=content, tool_calls=tool_calls)
    # end def invoke_messages
# end class ChatFireworks


def _parse_tool_calls(message: dict[str, Any]) -> list[ToolCall]:
    """ parse tool calls.

    Parameters
    ----------
    message : dict[str, Any]
        Argument value.

    Returns
    -------
    list[ToolCall]
        Returned value.
    """
    tool_calls: list[ToolCall] = []
    calls = message.get("tool_calls")
    if not isinstance(calls, list):
        return tool_calls
    # end if

    for call in calls:
        if not isinstance(call, dict):
            continue
        # end if
        call_id = str(call.get("id", ""))
        function = call.get("function") if isinstance(call.get("function"), dict) else {}
        name = str(function.get("name", ""))
        raw_arguments = function.get("arguments", {})
        if isinstance(raw_arguments, str):
            try:
                args: Any = json.loads(raw_arguments)
            except Exception:
                args = raw_arguments
            # end try
        else:
            args = raw_arguments
        # end if
        if name:
            tool_calls.append(ToolCall(id=call_id, name=name, args=args))
        # end if
    # end for
    return tool_calls
# end def _parse_tool_calls
