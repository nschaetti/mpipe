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

"""mpipe.commands.prompt module.

Notes
-----
This module is part of the Python port of the `mpipe` project.
"""
from __future__ import annotations

import click

from mpipe.commands.prompting import build_messages, compose_prompt, non_empty, resolve_prompt
from mpipe.console import console, print_json


@click.group("prompt")
def prompt_group() -> None:
    """Prompt group.

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
# end def prompt_group


@prompt_group.command("render")
@click.option("--system")
@click.option("--prompt")
@click.option("--postprompt")
@click.option("--json", "json_output", is_flag=True)
@click.argument("input_prompt", required=False)
def prompt_render_command(
    system: str | None,
    prompt: str | None,
    postprompt: str | None,
    json_output: bool,
    input_prompt: str | None,
) -> None:
    """Prompt render command.

    Parameters
    ----------
    system : str | None
        Argument value.
    prompt : str | None
        Argument value.
    postprompt : str | None
        Argument value.
    json_output : bool
        Argument value.
    input_prompt : str | None
        Argument value.

    Returns
    -------
    None
        Returned value.
    """
    try:
        main_prompt = resolve_prompt(input_prompt)
        rendered_prompt = compose_prompt(prompt, main_prompt.text, postprompt)
        messages = build_messages(non_empty(system), rendered_prompt)
        if json_output:
            print_json(
                {
                    "prompt": rendered_prompt,
                    "messages": [message.to_json() for message in messages],
                    "prompt_source": main_prompt.source.as_str(),
                }
            )
            return
        # end if
        console.print(rendered_prompt)
    except Exception as err:
        raise click.ClickException(str(err)) from err
    # end try
# end def prompt_render_command
