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

"""mpipe.commands.prompting module.

Notes
-----
This module is part of the Python port of the `mpipe` project.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from enum import Enum

from mpipe.rchain.provider import ChatMessage, MessageContent


@dataclass(slots=True)
class PromptInput:
    """Promptinput.

    Notes
    -----
    This class follows the same role and hierarchy as the Rust implementation.
    """
    text: str
    source: "PromptSource"
# end class PromptInput


class PromptSource(str, Enum):
    """Promptsource.

    Notes
    -----
    This class follows the same role and hierarchy as the Rust implementation.
    """
    ARGUMENT = "argument"
    FILE = "file"
    STDIN = "stdin"

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
# end class PromptSource


def non_empty(value: str | None) -> str | None:
    """Non empty.

    Parameters
    ----------
    value : str | None
        Argument value.

    Returns
    -------
    str | None
        Returned value.
    """
    if value is None:
        return None
    # end if
    trimmed = value.strip()
    return trimmed or None
# end def non_empty


def build_messages(system: str | None, prompt: str) -> list[ChatMessage]:
    """Build messages.

    Parameters
    ----------
    system : str | None
        Argument value.
    prompt : str
        Argument value.

    Returns
    -------
    list[ChatMessage]
        Returned value.
    """
    messages: list[ChatMessage] = []
    if system is not None:
        messages.append(ChatMessage.system(MessageContent.text(system)))
    # end if
    messages.append(ChatMessage.user(MessageContent.text(prompt)))
    return messages
# end def build_messages


def build_messages_with_image(system: str | None, prompt: str, image_url: str) -> list[ChatMessage]:
    """Build messages with image.

    Parameters
    ----------
    system : str | None
        Argument value.
    prompt : str
        Argument value.
    image_url : str
        Argument value.

    Returns
    -------
    list[ChatMessage]
        Returned value.
    """
    messages: list[ChatMessage] = []
    if system is not None:
        messages.append(ChatMessage.system(MessageContent.text(system)))
    # end if
    messages.append(ChatMessage.user_with_text_and_image(prompt, image_url))
    return messages
# end def build_messages_with_image


def compose_prompt(preprompt: str | None, main_prompt: str, postprompt: str | None) -> str:
    """Compose prompt.

    Parameters
    ----------
    preprompt : str | None
        Argument value.
    main_prompt : str
        Argument value.
    postprompt : str | None
        Argument value.

    Returns
    -------
    str
        Returned value.
    """
    parts: list[str] = []
    if preprompt is not None and preprompt.strip():
        parts.append(preprompt)
    # end if
    parts.append(main_prompt)
    if postprompt is not None and postprompt.strip():
        parts.append(postprompt)
    # end if
    return "\n\n".join(parts)
# end def compose_prompt


def resolve_prompt(cli_prompt: str | None) -> PromptInput:
    """Resolve prompt.

    Parameters
    ----------
    cli_prompt : str | None
        Argument value.

    Returns
    -------
    PromptInput
        Returned value.
    """
    if cli_prompt is not None:
        return PromptInput(text=cli_prompt, source=PromptSource.ARGUMENT)
    # end if

    if sys.stdin.isatty():
        raise ValueError("No prompt provided. Pass an argument or pipe stdin.")
    # end if

    text = sys.stdin.read().strip()
    if not text:
        raise ValueError("Prompt is empty.")
    # end if
    return PromptInput(text=text, source=PromptSource.STDIN)
# end def resolve_prompt
