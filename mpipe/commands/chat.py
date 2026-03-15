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

from mpipe.commands._helpers import _json_line, render_version, resolve_profile
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
from mpipe.rchain.provider import AskOptions, ChatMessage, Provider


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
@click.option("--show-usage", is_flag=True)
@click.option("--verbose", is_flag=True)
@click.option("--fail-on-empty", is_flag=True)
@click.option("--system")
def chat_command(**kwargs: Any) -> None:
    """
    Chat command.
    """
    try:
        _run_chat(**kwargs)
    except Exception as err:
        err_console.print(str(err), style="red")
        raise SystemExit(1)
    # end try
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
    output: str | None,
    json_output: bool,
    show_usage: bool,
    quiet: bool,
    verbose: bool,
    dry_run: bool,
    fail_on_empty: bool,
    save: Path | None,
    system: str | None,
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
    output : str | None
        Argument value.
    json_output : bool
        Argument value.
    show_usage : bool
        Argument value.
    quiet : bool
        Argument value.
    verbose : bool
        Argument value.
    dry_run : bool
        Argument value.
    fail_on_empty : bool
        Argument value.
    save : Path | None
        Argument value.
    system : str | None
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
    output_format = resolve_output_format(output, json_output, profile_cfg)
    resolved_show_usage = resolve_show_usage(show_usage, profile_cfg)
    resolved_system = resolve_system(system, profile_cfg)

    options = AskOptions(
        temperature=resolved_temperature,
        max_tokens=resolved_max_tokens,
        timeout_secs=timeout_secs,
        retries=retry_count,
        retry_delay_ms=retry_delay_ms,
    )

    main_prompt = resolve_main_prompt(prompt, input_prompt, prompt_file)
    final_prompt = main_prompt.text

    if image is not None:
        resolved_url = provider.resolve_image_url(image)
        messages = build_messages_with_image(non_empty(resolved_system), final_prompt, resolved_url)
    else:
        messages = build_messages(non_empty(resolved_system), final_prompt)
    # end if

    if verbose and not quiet:
        log_verbose(
            selected_provider,
            selected_model,
            output_format,
            dry_run,
            resolved_show_usage,
            main_prompt.source,
            messages,
            options,
        )
    # end if

    if dry_run:
        payload = {
            "dry_run": True,
            "provider": selected_provider.as_str(),
            "endpoint": provider.endpoint(selected_provider),
            "model": selected_model,
            "messages": [message.to_json() for message in messages],
            "request": {
                "temperature": resolved_temperature,
                "max_tokens": resolved_max_tokens,
                "timeout_secs": timeout_secs,
                "retries": retry_count,
                "retry_delay_ms": retry_delay_ms,
            },
            "output": output_format,
            "show_usage": resolved_show_usage,
            "authorization": "Bearer ***REDACTED***",
        }
        rendered = _json_line(payload)
        console.print(rendered, markup=False)
        if save is not None:
            write_output(save, rendered + "\n")
        # end if
        if resolved_show_usage and not quiet:
            err_console.print("usage: unavailable latency_ms=0 (dry-run)")
        # end if
        return
    # end if

    start = time.perf_counter()
    response = asyncio.run(provider.ask(selected_provider, selected_model, messages, options))
    latency_ms = int((time.perf_counter() - start) * 1000)

    if fail_on_empty and not response.content.strip():
        raise ValueError("Model response is empty and --fail-on-empty is enabled.")
    # end if

    usage = None
    if response.usage is not None:
        usage = UsageData(
            prompt_tokens=response.usage.prompt_tokens,
            completion_tokens=response.usage.completion_tokens,
            total_tokens=response.usage.total_tokens,
        )
    # end if

    if resolved_show_usage and not quiet:
        print_usage(usage, latency_ms)
    # end if

    if output_format == "text":
        rendered = response.content
        console.print(rendered, markup=False, end="")
    else:
        payload = {
            "provider": selected_provider.as_str(),
            "model": selected_model,
            "answer": response.content,
            "latency_ms": latency_ms,
            "request": {
                "temperature": resolved_temperature,
                "max_tokens": resolved_max_tokens,
                "timeout_secs": timeout_secs,
                "retries": retry_count,
                "retry_delay_ms": retry_delay_ms,
            },
            "usage": json_usage(usage),
        }
        rendered = _json_line(payload)
        console.print(rendered, markup=False)
    # end if

    if save is not None:
        write_output(save, rendered + ("\n" if not rendered.endswith("\n") else ""))
    # end if
# end def _run_chat

