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


from __future__ import annotations

import asyncio
import os
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import click

from mpipe.commands._helpers import _json_line, render_version
from mpipe.commands.ask import UsageData
from mpipe.commands.config import (
    resolve_profile,
    resolve_retries,
    resolve_model,
    resolve_provider,
    resolve_timeout,
    resolve_system,
    resolve_temperature,
    resolve_max_tokens,
    resolve_output_format,
    resolve_show_usage,
    resolve_retry_delay,
)
from mpipe.logging import LogConfig, LogLevels
from mpipe.commands.prompting import (
    PromptInput,
    PromptSource,
    build_messages,
    build_messages_with_image,
    compose_prompt,
    non_empty,
    resolve_prompt,
)
from mpipe.config import ProfileConfig, load_profile
from mpipe.console import console, err_console, print_json
from mpipe.rchain import provider
from mpipe.rchain.provider import AskOptions, ChatMessage, Provider, MessageContent


@click.command("chat")
@click.option("--version", "show_version", is_flag=True)
@click.option("--profile")
@click.option("--provider", "provider_name", type=click.Choice(["openai", "fireworks"]))
@click.option("--model")
@click.option("--temperature", type=float)
@click.option("--max-tokens", type=int)
@click.option("--timeout", type=int)
@click.option("--retries", type=int)
@click.option("--retry-delay", type=int)
@click.option("--fail-on-empty", is_flag=True)
@click.option("-v", "--verbose", count=True, help="Verbosity level")
def chat_command(**kwargs: Any) -> None:
    """
    Chat command.
    """
    _run_chat(**kwargs)
# end def chat_command


def _run_chat(
        show_version: bool,
        profile: str | None,
        provider_name: str | None,
        model: str | None,
        temperature: float | None,
        max_tokens: int | None,
        timeout: int | None,
        retries: int | None,
        retry_delay: int | None,
        fail_on_empty: bool,
        verbose: int = 0,
) -> None:
    """ run ask.

    Parameters
    ----------
    show_version : bool
        Argument value.
    profile : str | None
        Argument value.
    provider_name : str | None
        Argument value.
    model : str | None
        Argument value.
    temperature : float | None
        Argument value.
    max_tokens : int | None
        Argument value.
    timeout : int | None
        Argument value.
    retries : int | None
        Argument value.
    retry_delay : int | None
        Argument value.
    verbose : int
        Argument value.

    Returns
    -------
    None
        Returned value.
    """
    if show_version:
        console.print(render_version())
        return
    # end if

    profile_cfg = resolve_profile(profile)
    selected_provider = resolve_provider(provider_name, profile_cfg)
    selected_model = resolve_model(model, profile_cfg)
    resolved_temperature = resolve_temperature(temperature, profile_cfg)
    resolved_max_tokens = resolve_max_tokens(max_tokens, profile_cfg)
    timeout_secs = resolve_timeout(timeout, profile_cfg)
    retry_count = resolve_retries(retries, profile_cfg)
    retry_delay_ms = resolve_retry_delay(retry_delay, profile_cfg)

    # Verbosity options
    log_options = LogConfig(
        level=LogLevels(verbose)
    )

    # Ask options
    options = AskOptions(
        temperature=resolved_temperature,
        max_tokens=resolved_max_tokens,
        timeout_secs=timeout_secs,
        retries=retry_count,
        retry_delay_ms=retry_delay_ms,
    )

    # Chat on
    chat_on = True

    # List of messages
    messages: list[ChatMessage] = []

    while chat_on:
        # Ask prompt
        print(">> ", end="")
        chat_input = input()

        # Add messages
        messages.append(ChatMessage(role="user", content=MessageContent.text(chat_input)))

        # Send message to LLM
        response = asyncio.run(provider.ask(selected_provider, selected_model, messages, options, log_options))

        if log_options.level.value >= LogLevels.DEBUG.value:
            console.print(f"Fireworks API response: {response}")
        # end if

        if fail_on_empty and not response.content.strip():
            raise ValueError("Model response is empty and --fail-on-empty is enabled.")
        # end if

        if response.usage is not None:
            usage = UsageData(
                prompt_tokens=response.usage.prompt_tokens,
                completion_tokens=response.usage.completion_tokens,
                total_tokens=response.usage.total_tokens,
            )
        # end if

        for choice in response.choices:
            if choice.message.reasoning_content:
                console.print(f"<Thinking>\n{choice.message.reasoning_content}\n</Thinking>\n", markup=True)
            # end if
            rendered = choice.message.content
            console.print(rendered, markup=True)
        # end for
    # end while
# end def _run_chat
