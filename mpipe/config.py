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

"""mpipe.config module.

Notes
-----
This module is part of the Python port of the `mpipe` project.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    import tomllib  # py311+
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib  # type: ignore[no-redef]
# end try


@dataclass(slots=True)
class ProfileConfig:
    """Profileconfig.

    Notes
    -----
    This class follows the same role and hierarchy as the Rust implementation.
    """
    provider: str | None = None
    model: str | None = None
    system: str | None = None
    temperature: float | None = None
    max_tokens: int | None = None
    timeout: int | None = None
    retries: int | None = None
    retry_delay: int | None = None
    output: str | None = None
    show_usage: bool | None = None
    embedding_model: str | None = None
    chunk_size: int | None = None
    chunk_overlap: int | None = None
    chunk_strategy: str | None = None
# end class ProfileConfig


@dataclass(slots=True)
class ProviderDefaultsConfig:
    """Providerdefaultsconfig.

    Notes
    -----
    This class follows the same role and hierarchy as the Rust implementation.
    """
    model: str | None = None
    system: str | None = None
    temperature: float | None = None
    max_tokens: int | None = None
    timeout: int | None = None
    retries: int | None = None
    retry_delay: int | None = None
    output: str | None = None
    show_usage: bool | None = None
# end class ProviderDefaultsConfig


def load_profile(name: str) -> ProfileConfig:
    """Load profile.

    Parameters
    ----------
    name : str
        Argument value.

    Returns
    -------
    ProfileConfig
        Returned value.
    """
    path, config = _load_and_validate_config_file()
    profiles = config.get("profiles")
    if not isinstance(profiles, dict):
        raise ValueError(f"Config file '{path}' does not contain a [profiles] section.")
    # end if

    raw_profile = profiles.get(name)
    if not isinstance(raw_profile, dict):
        raise ValueError(f"Profile '{name}' not found in config file '{path}'.")
    # end if
    profile = _profile_from_dict(raw_profile)

    defaults = None
    provider = _normalized_provider_value(profile.provider or "")
    if provider:
        defaults = _provider_defaults_for(config, provider)
    # end if

    return _merge_provider_defaults(defaults, profile)
# end def load_profile


def validate_config(profile_name: str | None) -> Path:
    """Validate config.

    Parameters
    ----------
    profile_name : str | None
        Argument value.

    Returns
    -------
    Path
        Returned value.
    """
    path, config = _load_and_validate_config_file()
    if profile_name:
        profiles = config.get("profiles")
        if not isinstance(profiles, dict) or profile_name not in profiles:
            raise ValueError(f"Profile '{profile_name}' not found in config file '{path}'.")
        # end if
    # end if
    return path
# end def validate_config


def _load_and_validate_config_file() -> tuple[Path, dict[str, Any]]:
    """ load and validate config file.

    Parameters
    ----------
    None
        This callable does not accept explicit parameters.

    Returns
    -------
    tuple[Path, dict[str, Any]]
        Returned value.
    """
    path = config_path()
    try:
        raw = path.read_bytes()
    except OSError as err:
        raise ValueError(f"Failed to read config file '{path}': {err}") from err
    # end try

    try:
        config = tomllib.loads(raw.decode("utf-8"))
    except Exception as err:
        raise ValueError(f"Failed to parse config file '{path}': {err}") from err
    # end try

    if not isinstance(config, dict):
        raise ValueError(f"Failed to parse config file '{path}': root must be an object.")
    # end if

    _validate_config_file(config, path)
    return path, config
# end def _load_and_validate_config_file


def _provider_defaults_for(config: dict[str, Any], provider: str) -> ProviderDefaultsConfig | None:
    """ provider defaults for.

    Parameters
    ----------
    config : dict[str, Any]
        Argument value.
    provider : str
        Argument value.

    Returns
    -------
    ProviderDefaultsConfig | None
        Returned value.
    """
    providers = config.get("providers")
    if not isinstance(providers, dict):
        return None
    # end if

    for name, section in providers.items():
        normalized = _normalized_provider_value(name)
        if normalized != provider or not isinstance(section, dict):
            continue
        # end if
        defaults = section.get("defaults")
        if isinstance(defaults, dict):
            return _provider_defaults_from_dict(defaults)
        # end if
    # end for
    return None
# end def _provider_defaults_for


def _merge_provider_defaults(defaults: ProviderDefaultsConfig | None, profile: ProfileConfig) -> ProfileConfig:
    """ merge provider defaults.

    Parameters
    ----------
    defaults : ProviderDefaultsConfig | None
        Argument value.
    profile : ProfileConfig
        Argument value.

    Returns
    -------
    ProfileConfig
        Returned value.
    """
    d = defaults or ProviderDefaultsConfig()
    return ProfileConfig(
        provider=profile.provider,
        model=profile.model or d.model,
        system=profile.system or d.system,
        temperature=profile.temperature if profile.temperature is not None else d.temperature,
        max_tokens=profile.max_tokens if profile.max_tokens is not None else d.max_tokens,
        timeout=profile.timeout if profile.timeout is not None else d.timeout,
        retries=profile.retries if profile.retries is not None else d.retries,
        retry_delay=profile.retry_delay if profile.retry_delay is not None else d.retry_delay,
        output=profile.output or d.output,
        show_usage=profile.show_usage if profile.show_usage is not None else d.show_usage,
        embedding_model=profile.embedding_model,
        chunk_size=profile.chunk_size,
        chunk_overlap=profile.chunk_overlap,
        chunk_strategy=profile.chunk_strategy,
    )
# end def _merge_provider_defaults


def _validate_config_file(config: dict[str, Any], path: Path) -> None:
    """ validate config file.

    Parameters
    ----------
    config : dict[str, Any]
        Argument value.
    path : Path
        Argument value.

    Returns
    -------
    None
        Returned value.
    """
    providers = config.get("providers")
    if isinstance(providers, dict):
        for provider_name, provider_section in providers.items():
            normalized = _normalized_provider_value(provider_name)
            if normalized is None:
                raise ValueError(
                    f"Invalid provider section 'providers.{provider_name}' in config file '{path}'. Supported values: openai, fireworks."
                )
            # end if
            defaults = provider_section.get("defaults") if isinstance(provider_section, dict) else None
            if isinstance(defaults, dict):
                _validate_profile_fields(path, f"providers.{normalized}.defaults", defaults)
            # end if
        # end for
    # end if

    profiles = config.get("profiles")
    if isinstance(profiles, dict):
        for name, raw_profile in profiles.items():
            if not isinstance(raw_profile, dict):
                continue
            # end if
            provider_raw = raw_profile.get("provider")
            if isinstance(provider_raw, str) and _normalized_provider_value(provider_raw) is None:
                provider = provider_raw.strip().lower()
                raise ValueError(
                    f"Invalid profile provider '{provider}'. Supported values: openai, fireworks. (at 'profiles.{name}.provider' in '{path}')"
                )
            # end if
            _validate_profile_fields(path, f"profiles.{name}", raw_profile)
        # end for
    # end if
# end def _validate_config_file


def _validate_profile_fields(path: Path, section_path: str, fields: dict[str, Any]) -> None:
    """ validate profile fields.

    Parameters
    ----------
    path : Path
        Argument value.
    section_path : str
        Argument value.
    fields : dict[str, Any]
        Argument value.

    Returns
    -------
    None
        Returned value.
    """
    temperature = fields.get("temperature")
    if temperature is not None and not (0.0 <= float(temperature) <= 2.0):
        raise ValueError(
            f"Invalid value at '{section_path}.temperature' in config file '{path}': {temperature} (must be in [0.0, 2.0])."
        )
    # end if

    max_tokens = fields.get("max_tokens")
    if max_tokens is not None and int(max_tokens) == 0:
        raise ValueError(
            f"Invalid value at '{section_path}.max_tokens' in config file '{path}': 0 (must be > 0)."
        )
    # end if

    timeout = fields.get("timeout")
    if timeout is not None and int(timeout) == 0:
        raise ValueError(
            f"Invalid value at '{section_path}.timeout' in config file '{path}': 0 (must be > 0)."
        )
    # end if

    retry_delay = fields.get("retry_delay")
    if retry_delay is not None and int(retry_delay) == 0:
        raise ValueError(
            f"Invalid value at '{section_path}.retry_delay' in config file '{path}': 0 (must be > 0)."
        )
    # end if

    output = fields.get("output")
    if isinstance(output, str):
        normalized = output.strip().lower()
        if normalized not in {"text", "json"}:
            if section_path.startswith("profiles."):
                profile_name = section_path[len("profiles.") :]
                raise ValueError(
                    f"Invalid profile output '{normalized}'. Supported values: text, json. (at 'profiles.{profile_name}.output' in '{path}')"
                )
            # end if
            raise ValueError(
                f"Invalid value at '{section_path}.output' in config file '{path}': '{normalized}' (supported values: text, json)."
            )
        # end if
    # end if
# end def _validate_profile_fields


def _normalized_provider_value(raw: str) -> str | None:
    """ normalized provider value.

    Parameters
    ----------
    raw : str
        Argument value.

    Returns
    -------
    str | None
        Returned value.
    """
    value = raw.strip().lower()
    if value in {"openai", "fireworks"}:
        return value
    # end if
    return None
# end def _normalized_provider_value


def config_path() -> Path:
    """Config path.

    Parameters
    ----------
    None
        This callable does not accept explicit parameters.

    Returns
    -------
    Path
        Returned value.
    """
    mp_config = os.getenv("MP_CONFIG", "").strip()
    if mp_config:
        return Path(mp_config)
    # end if

    xdg = os.getenv("XDG_CONFIG_HOME", "").strip()
    if xdg:
        return Path(xdg) / "mpipe" / "config.toml"
    # end if

    home = os.getenv("HOME")
    if not home:
        raise ValueError("Cannot resolve config path: set MP_CONFIG or HOME/XDG_CONFIG_HOME.")
    # end if
    return Path(home) / ".config" / "mpipe" / "config.toml"
# end def config_path


def _profile_from_dict(data: dict[str, Any]) -> ProfileConfig:
    """ profile from dict.

    Parameters
    ----------
    data : dict[str, Any]
        Argument value.

    Returns
    -------
    ProfileConfig
        Returned value.
    """
    return ProfileConfig(
        provider=_get_str(data, "provider"),
        model=_get_str(data, "model"),
        system=_get_str(data, "system"),
        temperature=_get_float(data, "temperature"),
        max_tokens=_get_int(data, "max_tokens"),
        timeout=_get_int(data, "timeout"),
        retries=_get_int(data, "retries"),
        retry_delay=_get_int(data, "retry_delay"),
        output=_get_str(data, "output"),
        show_usage=_get_bool(data, "show_usage"),
        embedding_model=_get_str(data, "embedding_model"),
        chunk_size=_get_int(data, "chunk_size"),
        chunk_overlap=_get_int(data, "chunk_overlap"),
        chunk_strategy=_get_str(data, "chunk_strategy"),
    )
# end def _profile_from_dict


def _provider_defaults_from_dict(data: dict[str, Any]) -> ProviderDefaultsConfig:
    """ provider defaults from dict.

    Parameters
    ----------
    data : dict[str, Any]
        Argument value.

    Returns
    -------
    ProviderDefaultsConfig
        Returned value.
    """
    return ProviderDefaultsConfig(
        model=_get_str(data, "model"),
        system=_get_str(data, "system"),
        temperature=_get_float(data, "temperature"),
        max_tokens=_get_int(data, "max_tokens"),
        timeout=_get_int(data, "timeout"),
        retries=_get_int(data, "retries"),
        retry_delay=_get_int(data, "retry_delay"),
        output=_get_str(data, "output"),
        show_usage=_get_bool(data, "show_usage"),
    )
# end def _provider_defaults_from_dict


def _get_str(data: dict[str, Any], key: str) -> str | None:
    """ get str.

    Parameters
    ----------
    data : dict[str, Any]
        Argument value.
    key : str
        Argument value.

    Returns
    -------
    str | None
        Returned value.
    """
    value = data.get(key)
    return value if isinstance(value, str) else None
# end def _get_str


def _get_int(data: dict[str, Any], key: str) -> int | None:
    """ get int.

    Parameters
    ----------
    data : dict[str, Any]
        Argument value.
    key : str
        Argument value.

    Returns
    -------
    int | None
        Returned value.
    """
    value = data.get(key)
    return int(value) if isinstance(value, int) else None
# end def _get_int


def _get_float(data: dict[str, Any], key: str) -> float | None:
    """ get float.

    Parameters
    ----------
    data : dict[str, Any]
        Argument value.
    key : str
        Argument value.

    Returns
    -------
    float | None
        Returned value.
    """
    value = data.get(key)
    if isinstance(value, (int, float)):
        return float(value)
    # end if
    return None
# end def _get_float


def _get_bool(data: dict[str, Any], key: str) -> bool | None:
    """ get bool.

    Parameters
    ----------
    data : dict[str, Any]
        Argument value.
    key : str
        Argument value.

    Returns
    -------
    bool | None
        Returned value.
    """
    value = data.get(key)
    return value if isinstance(value, bool) else None
# end def _get_bool
