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


__all__ = ["cli", "tools_group", "tool_create_command"]

from mpipe.rchain.provider import ChatOptions, ChatMessage, Provider, ChatResponse, MessageContent

TOOL_COMMAND_EXTRACT = """You are a CLI help-to-tool parser.

Your task is to convert the raw output of a command's `--help` into a single STRICT JSON object with exactly two top-level keys:

- "tool": a function-calling tool definition
- "cli_map": a mapping used to reconstruct the CLI command

Hard requirements:
- Output JSON only.
- No markdown.
- No explanations.
- No text before or after the JSON.
- Use strict valid JSON, not Python dict syntax.
- Use double quotes for all keys and strings.
- Use true, false, null in JSON form.
- Never invent arguments, options, enums, default values, or subcommands not clearly supported by the help text.
- If something is unclear, use the most conservative interpretation.
- If a type is unclear, use "string".

You must return exactly one JSON object with this exact top-level structure:
{
  "tool": {
    "type": "function",
    "function": {
      "name": "tool_name_in_snake_case",
      "description": "short faithful description",
      "strict": true,
      "parameters": {
        "type": "object",
        "properties": {},
        "required": [],
        "additionalProperties": false
      }
    }
  },
  "cli_map": {}
}
No other top-level keys are allowed.

The output must contain exactly two top-level keys: "tool" and "cli_map".

"cli_map" must be a top-level sibling of "tool", not nested inside "tool" and not nested inside "tool.function".

Invalid:
{
  "tool": {
    ...,
    "cli_map": {}
  }
}

Valid:
{
  "tool": {},
  "cli_map": {}
}

Important parsing rules:
1. The "Usage:" line is the primary source of truth for positional arguments.
2. The "Arguments:" section is secondary and may refine positional arguments.
3. If "Arguments:" is empty or incomplete, still extract positional arguments from "Usage:".
4. Tokens like [OPTION], [OPTIONS], [OPTION]... are structural markers and must NOT become parameters.
5. Real placeholders like FILE, PATH, DIR, PATTERN, COMMAND, TARGET, QUERY, TEXT, etc. that appear in "Usage:" must become parameters.
6. If a positional appears as [FILE], it is optional.
7. If a positional appears as FILE, it is required.
8. If a positional appears as [FILE]... or FILE..., it is repeatable and must become an array.
9. Do not drop positional arguments just because their descriptions are missing.

Rules for "tool":
- "tool" must be suitable for function/tool calling.
- Use readable JSON parameter names in snake_case.
- Prefer long option names for parameter names.
- If only a short option exists, derive a readable snake_case name from its description.
- Positional arguments should also get readable snake_case names.
- Flags become boolean.
- Options with values become typed parameters.
- Repeatable arguments/options become arrays.
- Only include "enum" if the help explicitly gives a closed set of allowed values for that exact parameter.
- Ignore --help and --version.

Rules for "cli_map":
For every parameter present in tool.function.parameters.properties, there must be exactly one entry in "cli_map" with the same key.

Each cli_map entry must have:
- "kind": one of "positional", "flag", "option"

For positional arguments, use:
{
  "kind": "positional",
  "position": 0,
  "placeholder": "FILE",
  "repeatable": true
}

For boolean flags, use:
{
  "kind": "flag",
  "short": "-l",
  "long": "--long"
}

For options with values, use:
{
  "kind": "option",
  "short": "-T",
  "long": "--tabsize",
  "placeholder": "COLS",
  "repeatable": false,
  "value_mode": "separate"
}

Additional cli_map rules:
- "position" is zero-based.
- "placeholder" should preserve the original CLI placeholder when available, like FILE, PATH, COLS.
- "repeatable" is required for positional and option entries when applicable.
- "value_mode" should be:
  - "equals" for forms like --color=<when>
  - "separate" for forms like --ignore PATTERN or -T COLS
  - "either" only if the help clearly supports both
- Use null for missing short or long forms if needed.
- Do not invent aliases.

Type rules:
- boolean for flags
- integer for explicit integer placeholders like INT, N, NUM, COUNT, PORT, COLS
- number for explicit decimal/numeric placeholders like FLOAT, DOUBLE, RATIO, SECONDS
- string otherwise
- array for repeatable values

Example expectation:
If usage is:
Usage: ls [OPTION]... [FILE]...

Then:
- [OPTION]... does not become a parameter
- [FILE]... becomes an optional array parameter such as "files"
- cli_map["files"] must be:
  {
    "kind": "positional",
    "position": 0,
    "placeholder": "FILE",
    "repeatable": true
  }

Now parse the following help text and output JSON only:

{{HELP_TEXT}}
"""


TOOL_COMMAND_CREATE_ERROR_JSON = """Your previous response was invalid because it was not valid JSON.

Parser error:
{{PARSER_ERROR}}

Fix your previous answer and return it again.

Requirements:
- Return JSON only.
- No markdown.
- No explanation.
- No text before or after the JSON.
- The output must be parseable by Python's json.loads().
- Use double quotes for keys and strings.
- Use JSON literals: true, false, null.
- Keep the same intended content, but correct the formatting and structure.

Remember: the output must contain exactly two top-level keys:
- "tool"
- "cli_map"
"""


TOOL_COMMAND_CREATE_MISSING_TOOL = """Your previous response was invalid because the top-level key "tool" is missing.

Fix your previous answer and return it again.

Requirements:
- Return JSON only.
- No markdown.
- No explanation.
- No text before or after the JSON.
- The output must contain exactly two top-level keys:
  - "tool"
  - "cli_map"
- "tool" must be a top-level object, not nested inside another key.
- "cli_map" must remain a top-level sibling of "tool".

Expected top-level structure:
{
  "tool": {
    "type": "function",
    "function": {
      "name": "...",
      "description": "...",
      "strict": true,
      "parameters": {
        "type": "object",
        "properties": {},
        "required": [],
        "additionalProperties": false
      }
    }
  },
  "cli_map": {}
}
"""


TOOL_COMMAND_CREATE_MISSING_CLI_MAP = """Your previous response was invalid because the top-level key "cli_map" is missing.

Fix your previous answer and return it again.

Requirements:
- Return JSON only.
- No markdown.
- No explanation.
- No text before or after the JSON.
- The output must contain exactly two top-level keys:
  - "tool"
  - "cli_map"
- "cli_map" must be a top-level object, not nested inside "tool" and not nested inside "tool.function".
- "tool" and "cli_map" must be siblings at the top level.

Invalid example:
{
  "tool": {
    "...": "...",
    "cli_map": {}
  }
}

Valid shape:
{
  "tool": { ... },
  "cli_map": { ... }
}
"""


@click.group()
def cli():
    pass
# end def cli


# Group tools
@cli.group()
def tools_group():
    """Tools command group"""
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
    "--max-tokens",
    type=int,
    metavar="INT",
    help="Maximum generated tokens (> 0).",
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
    "--query-retries",
    type=int,
    default=4,
    metavar="INT",
    help="Number of retry attempts on transient failures.",
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
@click.option("--quiet", is_flag=True, help="Silence optional logs such as usage/verbose lines.")
@click.option("--verbose", is_flag=True, help="Print resolved request settings to stderr.")
def tool_create_command(**kwargs: Any) -> None:
    """Tools command"""
    _run_tool_create(**kwargs)
# end def tools_command


def _read_stdin() -> Optional[str]:
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
    """Enum for tool validation results."""
    SUCCESS = "success"
    INVALID_JSON = "invalid_json"
    MISSING_TOOL = "missing_tool"
    MISSING_CLI_MAP = "missing_cli_map"
# end class ToolValidation


@dataclasses.dataclass(frozen=True)
class _ToolValidationResult:
    """Result of tool validation."""
    result: _ToolValidationType
    error_message: Optional[str] = None
# end class _ToolValidationResult


def _validate_response(
        response_message: ChatMessage,
) -> _ToolValidationResult:
    """Validate the response message from the LLM tool.

    Args:
        response_message: The response message from the LLM tool.
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


def _run_tool_create(
        profile: str | None,
        provider_name: str | None,
        model: str | None,
        temperature: float | None,
        max_tokens: int | None,
        timeout: int | None,
        retries: int | None,
        retry_delay: int | None,
        query_retries: int | None,
        tool_name: str,
        tool_desc: str | None,
        quiet: bool,
        verbose: bool,
) -> None:
    """Run tools command."""
    profile_cfg = resolve_profile(profile)
    selected_provider = resolve_provider(provider_name, profile_cfg)
    selected_model = resolve_model(model, profile_cfg)
    resolved_temperature = resolve_temperature(temperature, profile_cfg)
    resolved_max_tokens = resolve_max_tokens(max_tokens, profile_cfg)
    timeout_secs = resolve_timeout(timeout, profile_cfg)
    retry_count = resolve_retries(retries, profile_cfg)
    retry_delay_ms = resolve_retry_delay(retry_delay, profile_cfg)

    # Chat options
    options = ChatOptions(
        temperature=resolved_temperature,
        max_tokens=resolved_max_tokens,
        timeout_secs=timeout_secs,
        retries=retry_count,
        retry_delay_ms=retry_delay_ms,
    )

    # Tool description from stdin
    tool_desc_in = _read_stdin()
    if tool_desc_in is None and tool_desc is None:
        console.print("[red bold]Error:[/] No tool description provided.", markup=True)
        return
    # end if

    # Compose prompt
    main_prompt = TOOL_COMMAND_EXTRACT.replace("{{HELP_TEXT}}", tool_desc_in or tool_desc or "")

    # Debug
    # console.print(f"[bold green]Tool description:[/bold green]: {main_prompt}", markup=True)

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

    query_count = 0
    query_success = False

    while query_count < query_retries:
        console.print(f"User: {messages[-1].content.value}")
        console.print("")

        start = time.perf_counter()
        response: ChatResponse = asyncio.run(provider.ask(selected_provider, selected_model, messages, options))
        latency_ms = int((time.perf_counter() - start) * 1000)

        # Get message
        response_message = response.get_message()

        if response_message is None:
            raise ValueError("Model response is empty.")
        # end if

        console.print(f"Model: {response_message.content.value}")
        console.print(f"Reasoning: {response_message.reasoning_content}")
        console.print("")

        # Validate response
        validation = _validate_response(response_message)

        if validation.result == _ToolValidationType.SUCCESS:
            query_success = True
            break
        elif validation.result == _ToolValidationType.INVALID_JSON:
            console.print(f"[red bold]Error[/]: Model response is not valid JSON", markup=True)
            response_error_message = TOOL_COMMAND_CREATE_ERROR_JSON.replace("{{PARSER_ERROR}}", validation.error_message or "")
        elif validation.result == _ToolValidationType.MISSING_TOOL:
            console.print(f"[red bold]Error[/]: Model response is missing the 'tool' key", markup=True)
            response_error_message = TOOL_COMMAND_CREATE_MISSING_TOOL
        elif validation.result == _ToolValidationType.MISSING_CLI_MAP:
            console.print(f"[red bold]Error[/]: Model response is missing the 'cli_map' key", markup=True)
            response_error_message = TOOL_COMMAND_CREATE_MISSING_CLI_MAP
        else:
            raise ValueError(f"Unknown validation result: {validation}")
        # end if

        # Add response to messages
        if query_count < query_retries - 1:
            messages.append(response.choices[0].message)
            messages.append(ChatMessage.user(MessageContent.text(response_error_message)))
        # end if

        query_count += 1
    # end while

    print(f"Last response: {json.loads(response_message.content.value)}")

    if query_success:
        console.print("[green bold]Success:[/] Tool created successfully.", markup=True)
    else:
        console.print(f"[red bold]Error:[/] Failed to create tool after {query_retries} attempts.", markup=True)
    # end if
# end def _run_tools
