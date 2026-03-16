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


from typing import Any
import os

from mpipe import __version__
from mpipe.config import ProfileConfig, load_profile
from mpipe.rchain.provider import Provider


def render_version() -> str:
    """Render version.

    Returns
    -------
    str
        Returned value.
    """
    commit = os.getenv("MP_GIT_SHA", "unknown")
    built = os.getenv("MP_BUILD_TS", "unknown")
    return f"{__version__}\ncommit: {commit}\nbuilt: {built}"
# end def render_version


def _json_line(payload: dict[str, Any]) -> str:
    """ json line.

    Parameters
    ----------
    payload : dict[str, Any]
        Argument value.

    Returns
    -------
    str
        Returned value.
    """
    import json
    return json.dumps(payload, ensure_ascii=False)
# end def _json_line


def parse_output_format(raw: str) -> str:
    """Parse output format.

    Parameters
    ----------
    raw : str
        Argument value.

    Returns
    -------
    str
        Returned value.
    """
    value = raw.strip().lower()
    if value in {"text", "json"}:
        return value
    # end if
    raise ValueError(f"Invalid profile output '{value}'. Supported values: text, json.")
# end def parse_output_format


def parse_provider_value(raw: str, source: str) -> Provider:
    """Parse provider value.

    Parameters
    ----------
    raw : str
        Argument value.
    source : str
        Argument value.

    Returns
    -------
    Provider
        Returned value.
    """
    value = raw.strip().lower()
    if value == "openai":
        return Provider.OPENAI
    # end if
    if value == "fireworks":
        return Provider.FIREWORKS
    # end if
    raise ValueError(f"Invalid {source} '{value}'. Supported values: openai, fireworks.")
# end def parse_provider_value

