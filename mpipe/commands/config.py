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

"""mpipe.commands.config module.

Notes
-----
This module is part of the Python port of the `mpipe` project.
"""
from __future__ import annotations

import os

import click

from mpipe import config as config_lib
from mpipe.commands._helpers import parse_provider_value, parse_output_format
from mpipe.config import ProfileConfig
from mpipe.console import console
from mpipe.rchain.provider import Provider


@click.group("config")
def config_group() -> None:
    """Config group.

    Parameters
    ----------
    None
        This callable does not accept explicit parameters.

    Returns
    -------
    None
        Returned value.
    """
    pass
# end def config_group


@config_group.command("check")
@click.option("--profile")
def config_check_command(profile: str | None) -> None:
    """Config check command.

    Parameters
    ----------
    profile : str | None
        Argument value.

    Returns
    -------
    None
        Returned value.
    """
    try:
        path = config_lib.validate_config(profile)
    except Exception as err:
        raise click.ClickException(str(err)) from err
    # end try
    console.print(f"config OK: {path}")
# end def config_check_command


def resolve_profile(profile_name: str | None) -> ProfileConfig:
    """Resolve profile.

    Parameters
    ----------
    profile_name : str | None
        Argument value.

    Returns
    -------
    ProfileConfig
        Returned value.
    """
    if profile_name:
        return config_lib.load_profile(profile_name)
    # end if
    return ProfileConfig()
# end def resolve_profile


def resolve_provider(cli_provider: str | None, profile: ProfileConfig) -> Provider:
    """Resolve provider.

    Parameters
    ----------
    cli_provider : str | None
        Argument value.
    profile : ProfileConfig
        Argument value.

    Returns
    -------
    Provider
        Returned value.
    """
    if cli_provider is not None:
        return Provider(cli_provider)
    # end if
    env_provider = os.getenv("MP_PROVIDER")
    if env_provider is not None:
        return parse_provider_value(env_provider, "MP_PROVIDER")
    # end if
    if profile.provider is not None:
        return parse_provider_value(profile.provider, "profile provider")
    # end if
    return Provider.OPENAI
# end def resolve_provider


def resolve_model(cli_model: str | None, profile: ProfileConfig) -> str:
    """Resolve model.

    Parameters
    ----------
    cli_model : str | None
        Argument value.
    profile : ProfileConfig
        Argument value.

    Returns
    -------
    str
        Returned value.
    """
    for candidate in [cli_model, os.getenv("MP_MODEL"), profile.model]:
        if candidate is None:
            continue
        # end if
        trimmed = candidate.strip()
        if trimmed:
            return trimmed
        # end if
    # end for
    raise ValueError("No model provided. Use --model or set MP_MODEL.")
# end def resolve_model


def resolve_temperature(cli_temperature: float | None, profile: ProfileConfig) -> float | None:
    """Resolve temperature.

    Parameters
    ----------
    cli_temperature : float | None
        Argument value.
    profile : ProfileConfig
        Argument value.

    Returns
    -------
    float | None
        Returned value.
    """
    value = cli_temperature
    if value is None:
        env_value = os.getenv("MP_TEMPERATURE")
        if env_value is not None:
            try:
                value = float(env_value.strip())
            except ValueError as err:
                raise ValueError(
                    f"Invalid MP_TEMPERATURE '{env_value}'. Must be a float in [0.0, 2.0]."
                ) from err
            # end try
        # end if
    # end if
    if value is None:
        value = profile.temperature
    # end if
    if value is not None and not (0.0 <= value <= 2.0):
        raise ValueError(f"Invalid temperature {value}. Must be in [0.0, 2.0].")
    # end if
    return value
# end def resolve_temperature


def resolve_max_tokens(cli_max_tokens: int | None, profile: ProfileConfig) -> int | None:
    """Resolve max tokens.

    Parameters
    ----------
    cli_max_tokens : int | None
        Argument value.
    profile : ProfileConfig
        Argument value.

    Returns
    -------
    int | None
        Returned value.
    """
    value = cli_max_tokens
    if value is None:
        env_value = os.getenv("MP_MAX_TOKENS")
        if env_value is not None:
            try:
                value = int(env_value.strip())
            except ValueError as err:
                raise ValueError(
                    f"Invalid MP_MAX_TOKENS '{env_value}'. Must be an integer > 0."
                ) from err
            # end try
        # end if
    # end if
    if value is None:
        value = profile.max_tokens
    # end if
    if value is not None and value == 0:
        raise ValueError("Invalid max tokens 0. Must be > 0.")
    # end if
    return value
# end def resolve_max_tokens


def resolve_timeout(cli_timeout: int | None, profile: ProfileConfig) -> int | None:
    """Resolve timeout.

    Parameters
    ----------
    cli_timeout : int | None
        Argument value.
    profile : ProfileConfig
        Argument value.

    Returns
    -------
    int | None
        Returned value.
    """
    value = cli_timeout
    if value is None:
        env_value = os.getenv("MP_TIMEOUT")
        if env_value is not None:
            try:
                value = int(env_value.strip())
            except ValueError as err:
                raise ValueError(
                    f"Invalid MP_TIMEOUT '{env_value}'. Must be an integer > 0."
                ) from err
            # end try
        # end if
    # end if
    if value is None:
        value = profile.timeout
    # end if
    if value is not None and value == 0:
        raise ValueError("Invalid timeout 0. Must be > 0 seconds.")
    # end if
    return value
# end def resolve_timeout


def resolve_retries(cli_retries: int | None, profile: ProfileConfig) -> int:
    """Resolve retries.

    Parameters
    ----------
    cli_retries : int | None
        Argument value.
    profile : ProfileConfig
        Argument value.

    Returns
    -------
    int
        Returned value.
    """
    if cli_retries is not None:
        return cli_retries
    # end if
    env_value = os.getenv("MP_RETRIES")
    if env_value is not None:
        try:
            return int(env_value.strip())
        except ValueError as err:
            raise ValueError(f"Invalid MP_RETRIES '{env_value}'. Must be an integer >= 0.") from err
        # end try
    # end if
    return profile.retries if profile.retries is not None else 0
# end def resolve_retries


def resolve_retry_delay(cli_retry_delay: int | None, profile: ProfileConfig) -> int:
    """Resolve retry delay.

    Parameters
    ----------
    cli_retry_delay : int | None
        Argument value.
    profile : ProfileConfig
        Argument value.

    Returns
    -------
    int
        Returned value.
    """
    if cli_retry_delay is not None:
        value = cli_retry_delay
    elif os.getenv("MP_RETRY_DELAY") is not None:
        raw = os.getenv("MP_RETRY_DELAY", "")
        try:
            value = int(raw.strip())
        except ValueError as err:
            raise ValueError(f"Invalid MP_RETRY_DELAY '{raw}'. Must be an integer > 0.") from err
        # end try
    else:
        value = profile.retry_delay if profile.retry_delay is not None else 500
    # end if
    if value == 0:
        raise ValueError("Invalid retry delay 0. Must be > 0 milliseconds.")
    # end if
    return value
# end def resolve_retry_delay


def resolve_output_format(output: str | None, json_output: bool, profile: ProfileConfig) -> str:
    """Resolve output format.

    Parameters
    ----------
    output : str | None
        Argument value.
    json_output : bool
        Argument value.
    profile : ProfileConfig
        Argument value.

    Returns
    -------
    str
        Returned value.
    """
    if json_output:
        return "json"
    # end if
    if output is not None:
        return output
    # end if
    if profile.output is not None:
        return parse_output_format(profile.output)
    # end if
    return "text"
# end def resolve_output_format


def resolve_show_usage(cli_show_usage: bool, profile: ProfileConfig) -> bool:
    """Resolve show usage.

    Parameters
    ----------
    cli_show_usage : bool
        Argument value.
    profile : ProfileConfig
        Argument value.

    Returns
    -------
    bool
        Returned value.
    """
    if cli_show_usage:
        return True
    # end if
    return bool(profile.show_usage)
# end def resolve_show_usage


def resolve_system(cli_system: str | None, profile: ProfileConfig) -> str | None:
    """Resolve system.

    Parameters
    ----------
    cli_system : str | None
        Argument value.
    profile : ProfileConfig
        Argument value.

    Returns
    -------
    str | None
        Returned value.
    """
    if cli_system is not None:
        return cli_system
    # end if
    return profile.system
# end def resolve_system

