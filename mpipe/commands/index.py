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

"""mpipe.commands.index module.

Notes
-----
This module is part of the Python port of the `mpipe` project.
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import click

from mpipe.commands.chroma import ChromaConnectArgs, connect
from mpipe.console import console
from mpipe.rchain.embeddings import EmbeddingProvider, embed_chunks_with_provider


DEFAULT_COLLECTION = "mpipe"
DEFAULT_CHUNK_SIZE = 1000
DEFAULT_CHUNK_OVERLAP = 200


@dataclass(slots=True)
class Chunk:
    """Chunk.

    Notes
    -----
    This class follows the same role and hierarchy as the Rust implementation.
    """
    text: str
    char_start: int
    char_end: int
# end class Chunk


@click.command("index")
@click.option("--file", "file_path", type=click.Path(path_type=Path))
@click.option("--document")
@click.option("--embedding-model")
@click.option("--chunk-size", type=int)
@click.option("--chunk-overlap", type=int)
@click.option("--collection")
@click.option("--chroma-url")
@click.option("--chroma-host")
@click.option("--chroma-port", type=int)
@click.option("--chroma-scheme")
@click.option("--chroma-path", type=click.Path(path_type=Path))
@click.option("--source")
@click.option("--id-prefix")
@click.option("--metadata", multiple=True)
@click.option("--metadata-json", type=click.Path(path_type=Path))
def index_command(**kwargs: Any) -> None:
    """Index command.

    Parameters
    ----------
    **kwargs : Any
        Argument value.

    Returns
    -------
    None
        Returned value.
    """
    try:
        _run_index(**kwargs)
    except Exception as err:
        raise click.ClickException(str(err)) from err
    # end try
# end def index_command


def _run_index(
    file_path: Path | None,
    document: str | None,
    embedding_model: str | None,
    chunk_size: int | None,
    chunk_overlap: int | None,
    collection: str | None,
    chroma_url: str | None,
    chroma_host: str | None,
    chroma_port: int | None,
    chroma_scheme: str | None,
    chroma_path: Path | None,
    source: str | None,
    id_prefix: str | None,
    metadata: tuple[str, ...],
    metadata_json: Path | None,
) -> None:
    """ run index.

    Parameters
    ----------
    file_path : Path | None
        Argument value.
    document : str | None
        Argument value.
    embedding_model : str | None
        Argument value.
    chunk_size : int | None
        Argument value.
    chunk_overlap : int | None
        Argument value.
    collection : str | None
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
    source : str | None
        Argument value.
    id_prefix : str | None
        Argument value.
    metadata : tuple[str, ...]
        Argument value.
    metadata_json : Path | None
        Argument value.

    Returns
    -------
    None
        Returned value.
    """
    if file_path and document:
        raise ValueError("Use either --file or --document, not both.")
    # end if
    if not file_path and not document:
        raise ValueError("Missing input: provide --file or --document.")
    # end if
    if document and not source:
        raise ValueError("--source is required when using --document.")
    # end if
    if source is not None and not source.strip():
        raise ValueError("--source cannot be empty.")
    # end if

    size = chunk_size if chunk_size is not None else DEFAULT_CHUNK_SIZE
    overlap = chunk_overlap if chunk_overlap is not None else DEFAULT_CHUNK_OVERLAP
    if size <= 0:
        raise ValueError("--chunk-size must be > 0")
    # end if
    if overlap >= size:
        raise ValueError("--chunk-overlap must be < --chunk-size")
    # end if

    full_document = _read_document(file_path, document)
    chunks = split_text(full_document, size, overlap)
    if not chunks:
        raise ValueError("Document is empty after trimming.")
    # end if

    stdin_embeddings = _read_embeddings_from_stdin()
    if stdin_embeddings is not None:
        _validate_embeddings_count(stdin_embeddings, len(chunks))
        _validate_embeddings_dimensions(stdin_embeddings)
        embeddings = stdin_embeddings
    else:
        if embedding_model is None:
            raise ValueError(
                "Missing --embedding-model (required when stdin embeddings are not provided)."
            )
        # end if
        vectors = embed_chunks_with_provider(
            EmbeddingProvider.FIREWORKS,
            embedding_model,
            [chunk.text for chunk in chunks],
        )
        embeddings = [[float(value) for value in vector] for vector in vectors]
    # end if

    source_value = source.strip() if source else str(file_path)
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
        col = client.get_or_create_collection(collection_name)
        base_metadata = _load_metadata_json(metadata_json)
        overrides = _parse_metadata_overrides(list(metadata))
        base_metadata.update(overrides)

        ids = _build_ids(id_prefix, file_path, len(chunks))
        docs = [chunk.text for chunk in chunks]
        metadatas = _build_chunk_metadatas(chunks, base_metadata, source_value, len(chunks))

        col.upsert(
            ids=ids,
            documents=docs,
            metadatas=metadatas,
            embeddings=embeddings,
        )
    finally:
        if guard is not None:
            guard.close()
    # end try
        # end if

    console.print(f"indexed {len(chunks)} chunks into collection '{collection_name}'")
# end def _run_index


def _read_document(file_path: Path | None, document: str | None) -> str:
    """ read document.

    Parameters
    ----------
    file_path : Path | None
        Argument value.
    document : str | None
        Argument value.

    Returns
    -------
    str
        Returned value.
    """
    if file_path is not None:
        content = file_path.read_text(encoding="utf-8")
        if not content.strip():
            raise ValueError(f"File '{file_path}' is empty.")
        # end if
        return content
    # end if
    if document is None or not document.strip():
        raise ValueError("Provided --document is empty.")
    # end if
    return document.strip()
# end def _read_document


def _read_embeddings_from_stdin() -> list[list[float]] | None:
    """ read embeddings from stdin.

    Parameters
    ----------
    None
        This callable does not accept explicit parameters.

    Returns
    -------
    list[list[float]] | None
        Returned value.
    """
    if sys.stdin.isatty():
        return None
    # end if
    buffer = sys.stdin.read()
    if not buffer.strip():
        raise ValueError("Stdin embeddings are empty.")
    # end if
    embeddings: list[list[float]] = []
    for line_idx, raw_line in enumerate(buffer.splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        # end if
        vector: list[float] = []
        for value_idx, value in enumerate(line.split(","), start=1):
            trimmed = value.strip()
            if not trimmed:
                continue
            # end if
            try:
                vector.append(float(trimmed))
            except ValueError as err:
                raise ValueError(
                    f"Invalid float at line {line_idx} position {value_idx}: '{trimmed}'"
                ) from err
            # end try
        # end for
        if not vector:
            raise ValueError(f"No floats parsed for embeddings line {line_idx}.")
        # end if
        embeddings.append(vector)
    # end for
    if not embeddings:
        raise ValueError("No embeddings parsed from stdin.")
    # end if
    return embeddings
# end def _read_embeddings_from_stdin


def _validate_embeddings_count(embeddings: list[list[float]], chunk_count: int) -> None:
    """ validate embeddings count.

    Parameters
    ----------
    embeddings : list[list[float]]
        Argument value.
    chunk_count : int
        Argument value.

    Returns
    -------
    None
        Returned value.
    """
    if len(embeddings) != chunk_count:
        raise ValueError(
            f"Embeddings count ({len(embeddings)}) does not match chunk count ({chunk_count})."
        )
    # end if
# end def _validate_embeddings_count


def _validate_embeddings_dimensions(embeddings: list[list[float]]) -> None:
    """ validate embeddings dimensions.

    Parameters
    ----------
    embeddings : list[list[float]]
        Argument value.

    Returns
    -------
    None
        Returned value.
    """
    if not embeddings:
        return
    # end if
    expected = len(embeddings[0])
    if expected == 0:
        raise ValueError("Embeddings cannot be empty vectors.")
    # end if
    for idx, vector in enumerate(embeddings):
        if len(vector) != expected:
            raise ValueError(
                f"Embedding dimension mismatch at index {idx} (expected {expected}, got {len(vector)})."
            )
        # end if
    # end for
# end def _validate_embeddings_dimensions


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
    if env_collection:
        return env_collection
    # end if
    return DEFAULT_COLLECTION
# end def _resolve_collection_name


def _load_metadata_json(path: Path | None) -> dict[str, Any]:
    """ load metadata json.

    Parameters
    ----------
    path : Path | None
        Argument value.

    Returns
    -------
    dict[str, Any]
        Returned value.
    """
    if path is None:
        return {}
    # end if
    import json

    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Metadata JSON '{path}' must be a JSON object.")
    # end if
    return data
# end def _load_metadata_json


def _parse_metadata_overrides(entries: list[str]) -> dict[str, Any]:
    """ parse metadata overrides.

    Parameters
    ----------
    entries : list[str]
        Argument value.

    Returns
    -------
    dict[str, Any]
        Returned value.
    """
    overrides: dict[str, Any] = {}
    for entry in entries:
        key, _, value = entry.partition("=")
        if not key.strip() or not value.strip():
            raise ValueError(f"Invalid metadata entry '{entry}'. Expected KEY=VALUE.")
        # end if
        overrides[key.strip()] = value.strip()
    # end for
    return overrides
# end def _parse_metadata_overrides


def _build_ids(id_prefix: str | None, file_path: Path | None, chunk_count: int) -> list[str]:
    """ build ids.

    Parameters
    ----------
    id_prefix : str | None
        Argument value.
    file_path : Path | None
        Argument value.
    chunk_count : int
        Argument value.

    Returns
    -------
    list[str]
        Returned value.
    """
    if id_prefix is not None:
        prefix = id_prefix.strip()
        if not prefix:
            raise ValueError("--id-prefix cannot be empty")
        # end if
    elif file_path is not None:
        prefix = file_path.name or "file"
    else:
        prefix = "document"
    # end if
    return [f"{prefix}-{index}" for index in range(chunk_count)]
# end def _build_ids


def _build_chunk_metadatas(
    chunks: list[Chunk],
    base: dict[str, Any],
    source: str,
    chunk_count: int,
) -> list[dict[str, Any]]:
    """ build chunk metadatas.

    Parameters
    ----------
    chunks : list[Chunk]
        Argument value.
    base : dict[str, Any]
        Argument value.
    source : str
        Argument value.
    chunk_count : int
        Argument value.

    Returns
    -------
    list[dict[str, Any]]
        Returned value.
    """
    output: list[dict[str, Any]] = []
    for idx, chunk in enumerate(chunks):
        metadata = dict(base)
        metadata["source"] = source
        metadata["chunk_index"] = idx
        metadata["chunk_count"] = chunk_count
        metadata["char_start"] = chunk.char_start
        metadata["char_end"] = chunk.char_end
        output.append(metadata)
    # end for
    return output
# end def _build_chunk_metadatas


def split_text(text: str, chunk_size: int, overlap: int) -> list[Chunk]:
    """Split text.

    Parameters
    ----------
    text : str
        Argument value.
    chunk_size : int
        Argument value.
    overlap : int
        Argument value.

    Returns
    -------
    list[Chunk]
        Returned value.
    """
    if not text.strip():
        return []
    # end if
    chars = list(text)
    total = len(chars)
    chunks: list[Chunk] = []
    start = 0
    while start < total:
        end = min(start + chunk_size, total)
        if end < total:
            for idx in range(end - 1, start, -1):
                if chars[idx].isspace():
                    end = idx
                    break
                # end if
            # end for
        # end if
        if end == start:
            end = min(start + chunk_size, total)
        # end if
        chunk_text = "".join(chars[start:end])
        chunks.append(Chunk(text=chunk_text, char_start=start, char_end=end))
        if end == total:
            break
        # end if
        next_start = max(0, end - overlap)
        start = end if next_start <= start else next_start
    # end while
    return chunks
# end def split_text
