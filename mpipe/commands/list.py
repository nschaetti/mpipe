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

"""mpipe.commands.list module.

Notes
-----
This module is part of the Python port of the `mpipe` project.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import click

from mpipe.commands.chroma import ChromaConnectArgs, connect
from mpipe.console import console, print_json


DEFAULT_COLLECTION = "mpipe"


@click.command("list")
@click.option("--collection")
@click.option("--limit", default=20, type=int)
@click.option("--offset", default=0, type=int)
@click.option("--json", "json_output", is_flag=True)
@click.option("--chroma-url")
@click.option("--chroma-host")
@click.option("--chroma-port", type=int)
@click.option("--chroma-scheme")
@click.option("--chroma-path", type=click.Path(path_type=Path))
def list_command(
    collection: str | None,
    limit: int,
    offset: int,
    json_output: bool,
    chroma_url: str | None,
    chroma_host: str | None,
    chroma_port: int | None,
    chroma_scheme: str | None,
    chroma_path: Path | None,
) -> None:
    """List command.

    Parameters
    ----------
    collection : str | None
        Argument value.
    limit : int
        Argument value.
    offset : int
        Argument value.
    json_output : bool
        Argument value.
    chroma_url : str | None
        Argument value.
    chroma_host : str | None
        Argument value.
    chroma_port : int | None
        Argument value.
    chroma_scheme : str | None
        Argument value.
    chroma_path : Path | None
        Argument value.

    Returns
    -------
    None
        Returned value.
    """
    try:
        if limit <= 0:
            raise ValueError("--limit must be > 0")
        # end if
        collection_name = _resolve_collection_name(collection)
        client, guard = connect(
            ChromaConnectArgs(
                chroma_url=chroma_url,
                chroma_host=chroma_host,
                chroma_port=chroma_port,
                chroma_scheme=chroma_scheme,
                chroma_path=chroma_path,
            )
        )
        try:
            col = client.get_collection(collection_name)
            result = col.get(limit=limit, offset=offset, include=["metadatas", "documents"])
        finally:
            if guard is not None:
                guard.close()
        # end try
            # end if

        entries = _collect_entries(result)
        if json_output:
            print_json(entries)
            return
        # end if

        if not entries:
            console.print(f"no entries in collection '{collection_name}'")
            return
        # end if
        for entry in entries:
            source = entry.get("source") or "unknown"
            chunk_idx = entry.get("chunk_index")
            chunk_count = entry.get("chunk_count")
            if chunk_idx is not None and chunk_count is not None:
                chunk = f"{int(chunk_idx) + 1}/{chunk_count}"
            else:
                chunk = "-"
            # end if
            preview = _compact_preview(entry.get("document") or "")
            console.print(f"{entry['id']}\tsource={source}\tchunk={chunk}\t{preview}")
        # end for
    except Exception as err:
        raise click.ClickException(str(err)) from err
    # end try
# end def list_command


def _resolve_collection_name(cli_collection: str | None) -> str:
    """ resolve collection name.

    Parameters
    ----------
    cli_collection : str | None
        Argument value.

    Returns
    -------
    str
        Returned value.
    """
    if cli_collection and cli_collection.strip():
        return cli_collection.strip()
    # end if
    env_collection = os.getenv("CHROMA_COLLECTION", "").strip()
    return env_collection or DEFAULT_COLLECTION
# end def _resolve_collection_name


def _collect_entries(result: dict[str, Any]) -> list[dict[str, Any]]:
    """ collect entries.

    Parameters
    ----------
    result : dict[str, Any]
        Argument value.

    Returns
    -------
    list[dict[str, Any]]
        Returned value.
    """
    ids = result.get("ids") or []
    metadatas = result.get("metadatas") or []
    documents = result.get("documents") or []
    output = []
    for idx, item_id in enumerate(ids):
        metadata = metadatas[idx] if idx < len(metadatas) else None
        doc = documents[idx] if idx < len(documents) else None
        output.append(
            {
                "id": item_id,
                "source": metadata.get("source") if isinstance(metadata, dict) else None,
                "chunk_index": metadata.get("chunk_index") if isinstance(metadata, dict) else None,
                "chunk_count": metadata.get("chunk_count") if isinstance(metadata, dict) else None,
                "document": doc,
            }
        )
    # end for
    return output
# end def _collect_entries


def _compact_preview(input_text: str) -> str:
    """ compact preview.

    Parameters
    ----------
    input_text : str
        Argument value.

    Returns
    -------
    str
        Returned value.
    """
    compact = " ".join(input_text.split())
    if len(compact) <= 120:
        return compact
    # end if
    return compact[:120] + "..."
# end def _compact_preview
