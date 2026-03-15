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

"""mpipe.commands.completion module.

Notes
-----
This module is part of the Python port of the `mpipe` project.
"""
from __future__ import annotations

import click
from click.shell_completion import get_completion_class

from mpipe.console import console


@click.command("completion")
@click.argument("shell", type=click.Choice(["bash", "zsh", "fish"]))
@click.pass_context
def completion_command(ctx: click.Context, shell: str) -> None:
    """Completion command.

    Parameters
    ----------
    ctx : click.Context
        Argument value.
    shell : str
        Argument value.

    Returns
    -------
    None
        Returned value.
    """
    command = ctx.find_root().command
    if command is None:
        raise click.ClickException("Failed to resolve root command for completion generation")
    # end if
    completion_class = get_completion_class(shell)
    complete_var = "_MPPIPE_COMPLETE"
    helper = completion_class(
        cli=command,
        ctx_args={},
        prog_name="mpipe",
        complete_var=complete_var,
    )
    console.print(helper.source(), markup=False, end="")
# end def completion_command
