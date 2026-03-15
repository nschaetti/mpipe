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

"""mpipe.commands.embed module.

Notes
-----
This module is part of the Python port of the `mpipe` project.
"""
from __future__ import annotations

import sys
import os
from pathlib import Path

import click

from mpipe.config import ProfileConfig, load_profile
from mpipe.console import console, print_json
from mpipe.rchain import embeddings
from mpipe.rchain.embeddings import ChunkStrategy, EmbeddingProvider, EmbeddingsConfig


@click.command("embed")
@click.option("--profile")
@click.option("--provider", type=click.Choice(["openai", "fireworks"]))
@click.option("--model")
@click.option("--chunk-size", type=int)
@click.option("--chunk-overlap", type=int)
@click.option("--chunk-strategy", type=click.Choice(["paragraph", "sentence", "token"]))
@click.option("--output", type=click.Choice(["text", "json"]))
@click.option("--json", "json_output", is_flag=True)
@click.option("--file", "file_path", type=click.Path(path_type=Path))
@click.argument("input_text", required=False)
def embed_command(
    profile: str | None,
    provider: str | None,
    model: str | None,
    chunk_size: int | None,
    chunk_overlap: int | None,
    chunk_strategy: str | None,
    output: str | None,
    json_output: bool,
    file_path: Path | None,
    input_text: str | None,
) -> None:
    """Embed command.

    Parameters
    ----------
    profile : str | None
        Argument value.
    provider : str | None
        Argument value.
    model : str | None
        Argument value.
    chunk_size : int | None
        Argument value.
    chunk_overlap : int | None
        Argument value.
    chunk_strategy : str | None
        Argument value.
    output : str | None
        Argument value.
    json_output : bool
        Argument value.
    file_path : Path | None
        Argument value.
    input_text : str | None
        Argument value.

    Returns
    -------
    None
        Returned value.
    """
    try:
        profile_cfg = load_profile(profile) if profile else ProfileConfig()
        emb_provider = _resolve_provider(provider, profile_cfg)
        emb_model = _resolve_model(model, profile_cfg)
        emb_chunk_size = _resolve_chunk_size(chunk_size, profile_cfg)
        emb_chunk_overlap = _resolve_chunk_overlap(chunk_overlap, profile_cfg)
        emb_chunk_strategy = _resolve_chunk_strategy(chunk_strategy, profile_cfg)
        output_format = "json" if json_output else (output or "text")
        text = _resolve_input(input_text, file_path)

        config = EmbeddingsConfig(
            provider=emb_provider,
            model=emb_model,
            chunk_size=emb_chunk_size,
            chunk_overlap=emb_chunk_overlap,
            chunk_strategy=emb_chunk_strategy,
        )
        result = embeddings.embed_texts(config, [text])

        if output_format == "json":
            print_json(
                {
                    "provider": result.provider,
                    "model": result.model,
                    "chunks": result.chunks,
                    "embeddings": result.embeddings,
                }
            )
            return
        # end if

        for vector in result.embeddings:
            console.print(",".join(str(value) for value in vector))
        # end for
    except Exception as err:
        raise click.ClickException(str(err)) from err
    # end try
# end def embed_command


def _resolve_provider(cli_provider: str | None, profile: ProfileConfig) -> EmbeddingProvider:
    """ resolve provider.

    Parameters
    ----------
    cli_provider : str | None
        Argument value.
    profile : ProfileConfig
        Argument value.

    Returns
    -------
    EmbeddingProvider
        Returned value.
    """
    raw = cli_provider or (profile.provider.strip() if profile.provider else "fireworks")
    return EmbeddingProvider(raw)
# end def _resolve_provider


def _resolve_model(cli_model: str | None, profile: ProfileConfig) -> str:
    """ resolve model.

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
    if cli_model and cli_model.strip():
        return cli_model.strip()
    # end if
    if profile.embedding_model and profile.embedding_model.strip():
        return profile.embedding_model.strip()
    # end if
    env_model = os.getenv("MP_MODEL")
    if env_model and env_model.strip():
        return env_model.strip()
    # end if
    return EmbeddingProvider.FIREWORKS.as_str()
# end def _resolve_model


def _resolve_chunk_size(cli_chunk_size: int | None, profile: ProfileConfig) -> int:
    """ resolve chunk size.

    Parameters
    ----------
    cli_chunk_size : int | None
        Argument value.
    profile : ProfileConfig
        Argument value.

    Returns
    -------
    int
        Returned value.
    """
    value = cli_chunk_size if cli_chunk_size is not None else profile.chunk_size
    if value is None:
        return 8000
    # end if
    if value <= 0:
        raise ValueError("Chunk size must be greater than 0.")
    # end if
    return value
# end def _resolve_chunk_size


def _resolve_chunk_overlap(cli_chunk_overlap: int | None, profile: ProfileConfig) -> int:
    """ resolve chunk overlap.

    Parameters
    ----------
    cli_chunk_overlap : int | None
        Argument value.
    profile : ProfileConfig
        Argument value.

    Returns
    -------
    int
        Returned value.
    """
    value = cli_chunk_overlap if cli_chunk_overlap is not None else profile.chunk_overlap
    if value is None:
        return 10
    # end if
    if value < 0 or value > 100:
        raise ValueError("Chunk overlap must be between 0 and 100.")
    # end if
    return value
# end def _resolve_chunk_overlap


def _resolve_chunk_strategy(cli_chunk_strategy: str | None, profile: ProfileConfig) -> ChunkStrategy:
    """ resolve chunk strategy.

    Parameters
    ----------
    cli_chunk_strategy : str | None
        Argument value.
    profile : ProfileConfig
        Argument value.

    Returns
    -------
    ChunkStrategy
        Returned value.
    """
    raw = cli_chunk_strategy or profile.chunk_strategy or "paragraph"
    strategy = ChunkStrategy.from_str(raw)
    if strategy is None:
        raise ValueError(f"Invalid chunk strategy '{raw}'. Supported values: paragraph, sentence, token.")
    # end if
    return strategy
# end def _resolve_chunk_strategy


def _resolve_input(cli_input: str | None, file_path: Path | None) -> str:
    """ resolve input.

    Parameters
    ----------
    cli_input : str | None
        Argument value.
    file_path : Path | None
        Argument value.

    Returns
    -------
    str
        Returned value.
    """
    if cli_input and cli_input.strip():
        return cli_input.strip()
    # end if
    if file_path is not None:
        return file_path.read_text(encoding="utf-8")
    # end if
    if sys.stdin.isatty():
        raise ValueError("No input provided. Pass an argument, --file, or pipe stdin.")
    # end if
    data = sys.stdin.read().strip()
    if not data:
        raise ValueError("Input is empty.")
    # end if
    return data
# end def _resolve_input
