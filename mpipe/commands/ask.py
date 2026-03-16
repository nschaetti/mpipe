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

"""mpipe.commands.ask module.

Notes
-----
This module is part of the Python port of the `mpipe` project.
"""
from __future__ import annotations

import asyncio
import os
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import click

from mpipe.commands._helpers import (
    _json_line,
    render_version,
)
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
from mpipe.commands.prompting import (
    PromptInput,
    PromptSource,
    build_messages,
    build_messages_with_image,
    non_empty,
    resolve_prompt,
)
from mpipe.config import ProfileConfig, load_profile
from mpipe.console import console, err_console, print_json
from mpipe.rchain import provider
from mpipe.rchain.provider import AskOptions, ChatMessage, Provider


@dataclass(slots=True)
class UsageData:
    """
    Usagedata.

    Notes
    -----
    This class follows the same role and hierarchy as the Rust implementation.
    """
    prompt_tokens: int | None
    completion_tokens: int | None
    total_tokens: int | None
# end class UsageData


@click.command("ask")
@click.option("--version", "show_version", is_flag=True)
@click.option("--profile")
@click.option("--provider", "provider_name", type=click.Choice(["openai", "fireworks"]))
@click.option("--model")
@click.option("--temperature", type=float)
@click.option("--max-tokens", type=int)
@click.option("--timeout", type=int)
@click.option("--retries", type=int)
@click.option("--retry-delay", type=int)
@click.option("--output", type=click.Choice(["text", "json"]))
@click.option("--json", "json_output", is_flag=True)
@click.option("--show-usage", is_flag=True)
@click.option("--quiet", is_flag=True)
@click.option("--verbose", is_flag=True)
@click.option("--dry-run", is_flag=True)
@click.option("--fail-on-empty", is_flag=True)
@click.option("--save", type=click.Path(path_type=Path))
@click.option("--system")
@click.option("--image")
@click.option("-p", "--prompt")
@click.option("--prompt-file", type=click.Path(exists=False, path_type=Path))
@click.argument("input_prompt", required=False)
def ask_command(**kwargs: Any) -> None:
    """Ask command.

    Parameters
    ----------
    **kwargs : Any
        Argument value.

    Returns
    -------
    None
        Returned value.
    """
    try:
        _run_ask(**kwargs)
    except Exception as err:
        err_console.print(str(err), style="red")
        raise SystemExit(1)
    # end try
# end def ask_command


def _run_ask(
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
    image: str | None,
    prompt: str | None,
    prompt_file: Path | None,
    input_prompt: str | None,
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
    image : str | None
        Argument value.
    prompt : str | None
        Argument value.
    prompt_file : Path | None
        Argument value.
    input_prompt : str | None
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
# end def _run_ask


def json_usage(usage: UsageData | None) -> dict[str, int | None] | None:
    """Json usage.

    Parameters
    ----------
    usage : UsageData | None
        Argument value.

    Returns
    -------
    dict[str, int | None] | None
        Returned value.
    """
    if usage is None:
        return None
    # end if
    if usage.prompt_tokens is None and usage.completion_tokens is None and usage.total_tokens is None:
        return None
    # end if
    return asdict(usage)
# end def json_usage


def print_usage(usage: UsageData | None, latency_ms: int) -> None:
    """Print usage.

    Parameters
    ----------
    usage : UsageData | None
        Argument value.
    latency_ms : int
        Argument value.

    Returns
    -------
    None
        Returned value.
    """
    j = json_usage(usage)
    if j is not None:
        err_console.print(
            "usage: "
            f"prompt_tokens={j['prompt_tokens'] if j['prompt_tokens'] is not None else 'n/a'} "
            f"completion_tokens={j['completion_tokens'] if j['completion_tokens'] is not None else 'n/a'} "
            f"total_tokens={j['total_tokens'] if j['total_tokens'] is not None else 'n/a'} "
            f"latency_ms={latency_ms}"
        )
        return
    # end if
    err_console.print(f"usage: unavailable latency_ms={latency_ms}")
# end def print_usage


def resolve_optional_segment(cli_value: str | None, file_path: Path | None, option_name: str) -> str | None:
    """Resolve optional segment.

    Parameters
    ----------
    cli_value : str | None
        Argument value.
    file_path : Path | None
        Argument value.
    option_name : str
        Argument value.

    Returns
    -------
    str | None
        Returned value.
    """
    if file_path is not None:
        return read_text_file(file_path, option_name)
    # end if
    return cli_value
# end def resolve_optional_segment


def resolve_main_prompt(
    cli_prompt: str | None,
    input_prompt: str | None,
    prompt_file: Path | None,
) -> PromptInput:
    """Resolve main prompt.

    Parameters
    ----------
    cli_prompt : str | None
        Argument value.
    input_prompt : str | None
        Argument value.
    prompt_file : Path | None
        Argument value.

    Returns
    -------
    PromptInput
        Returned value.
    """
    if prompt_file is not None:
        content = read_text_file(prompt_file, "--prompt-file")
        text = content.strip()
        if not text:
            raise ValueError(f"--prompt-file file '{prompt_file}' is empty.")
        # end if
        return PromptInput(text=text, source=PromptSource.FILE)
    # end if
    if cli_prompt is not None:
        return PromptInput(text=cli_prompt, source=PromptSource.ARGUMENT)
    # end if
    return resolve_prompt(input_prompt)
# end def resolve_main_prompt


def read_text_file(path: Path, option_name: str) -> str:
    """Read text file.

    Parameters
    ----------
    path : Path
        Argument value.
    option_name : str
        Argument value.

    Returns
    -------
    str
        Returned value.
    """
    try:
        return path.read_text(encoding="utf-8")
    except OSError as err:
        raise ValueError(f"Failed to read {option_name} '{path}': {err}") from err
    # end try
# end def read_text_file


def write_output(path: Path, content: str) -> None:
    """Write output.

    Parameters
    ----------
    path : Path
        Argument value.
    content : str
        Argument value.

    Returns
    -------
    None
        Returned value.
    """
    if path.parent and str(path.parent) not in {"", "."}:
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
        except OSError as err:
            raise ValueError(f"Failed to create output directory '{path.parent}': {err}") from err
        # end try
    # end if

    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.tmp.", dir=str(path.parent or Path(".")))
    os.close(fd)
    tmp_path = Path(tmp_name)
    try:
        tmp_path.write_text(content, encoding="utf-8")
        tmp_path.replace(path)
    except OSError as err:
        try:
            tmp_path.unlink(missing_ok=True)
        except OSError:
            pass
        # end try
        raise ValueError(f"Failed to replace output file '{path}': {err}") from err
    # end try
# end def write_output


def log_verbose(
    selected_provider: Provider,
    model: str,
    output_format: str,
    dry_run: bool,
    show_usage: bool,
    prompt_source: PromptSource,
    messages: list[ChatMessage],
    options: AskOptions,
) -> None:
    """Log verbose.

    Parameters
    ----------
    selected_provider : Provider
        Argument value.
    model : str
        Argument value.
    output_format : str
        Argument value.
    dry_run : bool
        Argument value.
    show_usage : bool
        Argument value.
    prompt_source : PromptSource
        Argument value.
    messages : list[ChatMessage]
        Argument value.
    options : AskOptions
        Argument value.

    Returns
    -------
    None
        Returned value.
    """
    total_chars = sum(message.content.text_len() for message in messages)
    err_console.print(
        "verbose: "
        f"provider={selected_provider.as_str()} "
        f"endpoint={provider.endpoint(selected_provider)} "
        f"model={model} "
        f"output={output_format} "
        f"dry_run={str(dry_run).lower()} "
        f"show_usage={str(show_usage).lower()} "
        f"prompt_source={prompt_source.as_str()} "
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
# end def log_verbose


def run_for_mpask(**kwargs: Any) -> None:
    """Run for mpask.

    Parameters
    ----------
    **kwargs : Any
        Argument value.

    Returns
    -------
    None
        Returned value.
    """
    _run_ask(**kwargs)
# end def run_for_mpask
