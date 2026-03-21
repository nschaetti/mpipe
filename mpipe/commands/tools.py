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

"""CLI commands for generating tool bundles from help text.

This module exposes the ``mpipe tools`` command group and the
``mpipe tools create`` workflow used to infer a structured tool definition and
CLI mapping from raw ``--help`` output.
"""

import asyncio
import dataclasses
import json
import sys
import time
from enum import Enum
from pathlib import Path
from typing import Any, Optional, List, Literal
import click
from rich.pretty import pprint

from mpipe.commands._helpers import render_version
from mpipe.commands.ask import UsageData
from mpipe.commands.prompting import build_messages
from mpipe.console import err_console, console
from mpipe.rchain import provider
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
from mpipe.utils import create_config

from ._prompts import *


__all__ = ["cli", "tools_group", "tool_create_command"]

from mpipe.rchain.provider import ChatOptions, ChatMessage, Provider, ChatResponse, MessageContent, StructuredOutputFormatJSON
from ..rchain.tools import ToolBundle


@click.group()
def cli():
    """Register the top-level CLI group for tool-related commands."""
    pass
# end def cli


# Group tools
@cli.group()
def tools_group():
    """Register the ``tools`` command subgroup."""
    pass
# end def tools


@tools_group.command("create")
@click.option(
    "--profile",
    metavar="NAME",
    help="Load profile from config (e.g. provider, model, defaults).",
)
@click.option(
    "--provider",
    "provider_name",
    type=click.Choice(["openai", "fireworks"]),
    help="Provider to call. Falls back to MP_PROVIDER/profile/default.",
)
@click.option(
    "--model",
    metavar="MODEL",
    help="Model ID (required unless set via MP_MODEL or profile).",
)
@click.option(
    "--temperature",
    type=float,
    metavar="FLOAT",
    help="Sampling temperature in [0.0, 2.0].",
)
@click.option(
    "--timeout",
    type=int,
    metavar="SECONDS",
    help="Request timeout in seconds (> 0).",
)
@click.option(
    "--retries",
    type=int,
    metavar="INT",
    help="Number of retry attempts on transient failures.",
)
@click.option(
    "--retry-delay",
    type=int,
    metavar="MS",
    help="Base retry delay in milliseconds (> 0, exponential backoff).",
)
@click.option(
    "--tool-name",
    required=True,
    metavar="TOOLNAME",
    help="Name of the tool to create.",
)
@click.option(
    "--tool-desc",
    required=False,
    metavar="TOOLDESC",
    help="Description of the tool to create.",
)
@click.option("--output", is_flag=True, help="Print tool bundle to stdout.")
@click.option("--quiet", is_flag=True, help="Silence optional logs such as usage/verbose lines.")
@click.option("--verbose", is_flag=True, help="Print resolved request settings to stderr.")
@click.option(
    "--config-path",
    type=click.Path(path_type=Path),
    default=Path.home() / ".config" / "mpipe",
    help="Path to configuration directory."
)
def tool_create_command(**kwargs: Any) -> None:
    """Run the ``tools create`` command entrypoint.

    Args:
        **kwargs: Parsed Click options forwarded to ``_run_tool_create``.
    """
    _run_tool_create(**kwargs)
# end def tools_command


def _read_stdin() -> Optional[str]:
    """Read non-interactive stdin content.

    Returns:
        The stripped stdin content when piped, or ``None`` when stdin is a TTY
        or empty.
    """
    if sys.stdin.isatty():
        return None
    else:
        data = sys.stdin.read()
        if data.strip() == "":
            return None
        # end if
        return data
    # end if
# end def _read_stdin


def _log_verbose(
    selected_provider: Provider,
    model: str,
    messages: list[ChatMessage],
    options: ChatOptions,
) -> None:
    """Print verbose diagnostics for the outgoing provider request.

    Args:
        selected_provider: Provider selected for the request.
        model: Model identifier sent to the provider.
        messages: Outgoing chat messages.
        options: Runtime chat options used for the request.
    """
    total_chars = sum(message.content.text_len() for message in messages)
    err_console.print(
        "verbose: "
        f"provider={selected_provider.as_str()} "
        f"endpoint={provider.endpoint(selected_provider)} "
        f"model={model} "
        f"messages={len(messages)} "
        f"chars={total_chars} "
        f"api_key_present={str(provider.is_api_key_present(selected_provider)).lower()}"
    )
    err_console.print(
        "verbose: "
        f"options temperature={options.temperature if options.temperature is not None else 'n/a'} "
        f"max_tokens={options.max_tokens if options.max_tokens is not None else 'n/a'} "
        f"timeout_secs={options.timeout_secs if options.timeout_secs is not None else 'n/a'} "
        f"retries={options.retries} retry_delay_ms={options.retry_delay_ms} backoff=exponential"
    )
# end def _log_verbose


class _ToolValidationType(Enum):
    """Validation outcomes for model-generated tool bundles."""
    SUCCESS = "success"
    INVALID_JSON = "invalid_json"
    MISSING_TOOL = "missing_tool"
    MISSING_CLI_MAP = "missing_cli_map"
# end class ToolValidation


@dataclasses.dataclass(frozen=True)
class _ToolValidationResult:
    """Validation result returned by ``_validate_response``.

    Attributes:
        result: Validation status enum value.
        error_message: Optional parser or validation error details.
    """
    result: _ToolValidationType
    error_message: Optional[str] = None
# end class _ToolValidationResult


def _validate_response(
        response_message: ChatMessage,
) -> _ToolValidationResult:
    """Validate the response message from the LLM tool.

    Args:
        response_message: The response message from the LLM tool.

    Returns:
        A validation result describing whether required top-level keys are
        present and JSON is syntactically valid.
    """
    # Response dict
    try:
        obj = json.loads(response_message.content.value)
    except json.JSONDecodeError as err:
        return _ToolValidationResult(result=_ToolValidationType.INVALID_JSON, error_message=str(err))
    # end try

    if "tool" not in obj:
        return _ToolValidationResult(result=_ToolValidationType.MISSING_TOOL)
    # end if

    if "cli_map" not in obj:
        return _ToolValidationResult(result=_ToolValidationType.MISSING_CLI_MAP)
    # end if

    return _ToolValidationResult(result=_ToolValidationType.SUCCESS)
# end def _validate_response


def _compose_prompt(
        base_prompt: str,
        replacements: dict[str, str],
) -> str:
    """Compose a prompt by replacing placeholder tokens.

    Args:
        base_prompt: Prompt template containing literal placeholders.
        replacements: Mapping from placeholder token to replacement value.

    Returns:
        The prompt with all requested literal replacements applied.
    """
    for key, value in replacements.items():
        base_prompt = base_prompt.replace(key, value)
    # end for
    return base_prompt
# end def _compose_prompt


def _run_tool_create(
        profile: str | None,
        provider_name: str | None,
        model: str | None,
        temperature: float | None,
        timeout: int | None,
        retries: int | None,
        retry_delay: int | None,
        tool_name: str,
        tool_desc: str | None,
        quiet: bool,
        verbose: bool,
        config_path: Path | None,
        output: bool = False,
) -> None:
    """Generate and persist a tool bundle from CLI help text.

    Args:
        profile: Optional profile name used to resolve defaults.
        provider_name: Optional provider override.
        model: Optional model override.
        temperature: Optional sampling temperature override.
        timeout: Optional timeout override in seconds.
        retries: Optional retry count override.
        retry_delay: Optional retry delay override in milliseconds.
        tool_name: Output tool bundle name.
        tool_desc: Optional tool description/help text when not piped via stdin.
        quiet: Whether to suppress informational output.
        verbose: Whether to print verbose request diagnostics.
        config_path: Base configuration path for output files.
        output: Whether to print the generated bundle to stdout.
    """
    # Create config paths
    tools_path = create_config(subdir="tools", path=config_path)

    # Get config
    profile_cfg = resolve_profile(profile)
    selected_provider = resolve_provider(provider_name, profile_cfg)
    selected_model = resolve_model(model, profile_cfg)
    resolved_temperature = resolve_temperature(temperature, profile_cfg)
    timeout_secs = resolve_timeout(timeout, profile_cfg)
    retry_count = resolve_retries(retries, profile_cfg)
    retry_delay_ms = resolve_retry_delay(retry_delay, profile_cfg)

    # Chat options
    options = ChatOptions(
        temperature=resolved_temperature,
        timeout_secs=timeout_secs,
        retries=retry_count,
        retry_delay_ms=retry_delay_ms,
    )

    # Structured output
    structured_output = StructuredOutputFormatJSON(
        name="ToolBundle",
        json_schema=ToolBundle.json_schema()
    )

    # Tool description from stdin
    tool_desc_in = _read_stdin()
    if tool_desc_in is None and tool_desc is None:
        console.print("[red bold]Error:[/] No tool description provided.", markup=True)
        return
    # end if

    # Compose prompt
    main_prompt = _compose_prompt(
        base_prompt=TOOL_COMMAND_EXTRACT,
        replacements={"{{HELP_TEXT}}": tool_desc_in or tool_desc or ""}
    )

    # Build messages
    messages: List[ChatMessage] = build_messages(None, main_prompt)
    if verbose and not quiet:
        _log_verbose(
            selected_provider=selected_provider,
            model=selected_model,
            messages=messages,
            options=options,
        )
    # end if

    start = time.perf_counter()
    response: ChatResponse = asyncio.run(
        provider.ask(
            provider=selected_provider,
            model=selected_model,
            messages=messages,
            structured_output=structured_output,
            options=options
        )
    )
    latency_ms = int((time.perf_counter() - start) * 1000)

    # Get message
    response_message: ChatMessage = response.get_message()

    # Get tool bundle
    bundle = ToolBundle.model_validate_json(json_data=response_message.content.value)

    # Save tool bundle
    bundle_path = Path(tools_path / f"{tool_name}.json")
    bundle_path.write_text(bundle.model_dump_json(indent=2))

    if not quiet and not output:
        console.print(f"[green bold]Tool bundle[/] saved to {bundle_path}", markup=True)
    # end if

    if output:
        console.print(bundle.model_dump_json(indent=2))
    # end if
# end def _run_tools
