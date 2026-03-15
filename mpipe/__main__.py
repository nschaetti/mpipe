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


"""mpipe.__main__ module.

Notes
-----
This module is part of the Python port of the `mpipe` project.
"""
from __future__ import annotations

import click

from mpipe.commands.agent import agent_command
from mpipe.commands.ask import ask_command
from mpipe.commands.chat import chat_command
from mpipe.commands.completion import completion_command
from mpipe.commands.config import config_group
from mpipe.commands.download import download_command
from mpipe.commands.embed import embed_command
from mpipe.commands.grep import grep_command
from mpipe.commands.index import index_command
from mpipe.commands.list import list_command
from mpipe.commands.models import models_command
from mpipe.commands.prompt import prompt_group


ROOT_HELP_EXAMPLES = """Examples:
  mpipe ask --provider fireworks --model accounts/fireworks/models/kimi-k2-instruct-0905 \"2+2?\"
  echo \"2+2?\" | mpipe ask --provider openai --model gpt-4o-mini
  mpipe models --provider fireworks
  mpipe prompt render --prompt \"You are concise\" \"Explain retries\"
  mpipe config check
  mpipe completion bash > ~/.local/share/bash-completion/completions/mpipe
"""


@click.group(help="Multi-provider LLM CLI tools", epilog=ROOT_HELP_EXAMPLES)
def cli() -> None:
    """Cli.

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
# end def cli


cli.add_command(ask_command)
cli.add_command(agent_command)
cli.add_command(models_command)
cli.add_command(index_command)
cli.add_command(grep_command)
cli.add_command(list_command)
cli.add_command(prompt_group)
cli.add_command(embed_command)
cli.add_command(download_command)
cli.add_command(config_group)
cli.add_command(completion_command)


def main() -> None:
    """Main.

    Parameters
    ----------
    None
        This callable does not accept explicit parameters.

    Returns
    -------
    None
        Returned value.
    """
    cli()
# end def main


if __name__ == "__main__":
    main()
# end if
