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

import click

from mpipe import config as config_lib
from mpipe.console import console


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
