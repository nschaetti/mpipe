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

"""mpipe.commands.download module.

Notes
-----
This module is part of the Python port of the `mpipe` project.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

import click

from mpipe.console import console, err_console


@click.command("download")
@click.argument("url")
@click.option("-o", "--output", type=click.Path(path_type=Path), required=True)
@click.option("--audio-only", is_flag=True)
@click.option("--format", "fmt", default="mp4")
@click.option("--quality")
@click.option("--verbose", is_flag=True)
@click.option("--timeout", default=600, type=int)
def download_command(
    url: str,
    output: Path,
    audio_only: bool,
    fmt: str,
    quality: str | None,
    verbose: bool,
    timeout: int,
) -> None:
    """Download command.

    Parameters
    ----------
    url : str
        Argument value.
    output : Path
        Argument value.
    audio_only : bool
        Argument value.
    fmt : str
        Argument value.
    quality : str | None
        Argument value.
    verbose : bool
        Argument value.
    timeout : int
        Argument value.

    Returns
    -------
    None
        Returned value.
    """
    del timeout
    if verbose:
        err_console.print(f"mpipe: downloading from {url}")
    # end if

    args = [
        "--output",
        str(output),
        "--no-part",
        "--no-clean-info-json",
    ]
    if audio_only:
        args.extend(["--extract-audio", "--audio-format", fmt])
    else:
        args.append("--format")
        args.append(f"bestvideo[height<={quality}]+bestaudio/best" if quality else "bestvideo+bestaudio/best")
        args.extend(["--merge-output-format", "mp4"])
    # end if
    args.append(url)

    if verbose:
        err_console.print(f"mpipe: running yt-dlp with args: {args}")
    # end if

    process = subprocess.run(["yt-dlp", *args], capture_output=True, text=True)
    if process.returncode != 0:
        raise click.ClickException(f"yt-dlp failed: {process.stderr}")
    # end if

    if verbose and process.stdout:
        err_console.print(f"mpipe: {process.stdout}")
    # end if
    if output.exists():
        console.print(str(output))
        return
    # end if
    raise click.ClickException(f"Output file not found: {output}")
# end def download_command
