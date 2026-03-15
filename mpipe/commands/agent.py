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

"""mpipe.commands.agent module.

Notes
-----
This module is part of the Python port of the `mpipe` project.
"""
from __future__ import annotations

import click


@click.command("agent")
@click.option("--profile")
@click.option("--provider", type=click.Choice(["openai", "fireworks"]))
@click.option("--model")
@click.option("-p", "--prompt")
@click.option("--prompt-file", type=click.Path(exists=False))
def agent_command(**kwargs: object) -> None:
    """Agent command.

    Parameters
    ----------
    **kwargs : object
        Argument value.

    Returns
    -------
    None
        Returned value.
    """
    del kwargs
    return
# end def agent_command
