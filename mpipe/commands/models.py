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

"""mpipe.commands.models module.

Notes
-----
This module is part of the Python port of the `mpipe` project.
"""
from __future__ import annotations

import click

from mpipe.console import console, print_json


MODEL_CATALOG = [
    {
        "provider": "fireworks",
        "id": "accounts/fireworks/models/kimi-k2-instruct-0905",
        "recommended": True,
    },
    {
        "provider": "fireworks",
        "id": "accounts/fireworks/models/minimax-m2p5",
        "recommended": True,
    },
    {"provider": "openai", "id": "gpt-4o-mini", "recommended": True},
]


@click.command("models")
@click.option("--provider", type=click.Choice(["openai", "fireworks"]))
@click.option("--json", "json_output", is_flag=True)
def models_command(provider: str | None, json_output: bool) -> None:
    """Models command.

    Parameters
    ----------
    provider : str | None
        Argument value.
    json_output : bool
        Argument value.

    Returns
    -------
    None
        Returned value.
    """
    models = sorted(MODEL_CATALOG, key=lambda entry: (entry["provider"], entry["id"]))
    if provider is not None:
        models = [entry for entry in models if entry["provider"] == provider]
    # end if

    if json_output:
        payload = [
            {
                "provider": entry["provider"],
                "id": entry["id"],
                "source": "local",
                "recommended": entry["recommended"],
            }
            for entry in models
        ]
        print_json(payload)
        return
    # end if

    for entry in models:
        console.print(f"{entry['provider']}\t{entry['id']}")
    # end for
# end def models_command
