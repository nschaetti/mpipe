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

"""mpipe.commands.grep module.

Notes
-----
This module is part of the Python port of the `mpipe` project.
"""
from __future__ import annotations

import asyncio
import os
from pathlib import Path
from typing import Any

import click

from mpipe.commands.chroma import ChromaConnectArgs, connect
from mpipe.commands.prompting import resolve_prompt
from mpipe.console import console, print_json
from mpipe.rchain.embeddings import EmbeddingProvider, embed_chunks_with_provider
from mpipe.rchain.provider import ChatOptions, ChatMessage, Provider, ask as provider_ask


DEFAULT_COLLECTION = "mpipe"


@click.command("grep")
@click.option("--collection")
@click.option("--embedding-model", required=True)
@click.option("--top-k", default=5, type=int)
@click.option("--provider", "provider_name", type=click.Choice(["openai", "fireworks"]))
@click.option("--model")
@click.option("--system")
@click.option("--temperature", type=float)
@click.option("--max-tokens", type=int)
@click.option("--timeout", type=int)
@click.option("--json", "json_output", is_flag=True)
@click.option("--chroma-url")
@click.option("--chroma-host")
@click.option("--chroma-port", type=int)
@click.option("--chroma-scheme")
@click.option("--chroma-path", type=click.Path(path_type=Path))
@click.argument("input_prompt", required=False)
def grep_command(**kwargs: Any) -> None:
    """Grep command.

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
        _run_grep(**kwargs)
    except Exception as err:
        raise click.ClickException(str(err)) from err
    # end try
# end def grep_command


def _run_grep(
    collection: str | None,
    embedding_model: str,
    top_k: int,
    provider_name: str | None,
    model: str | None,
    system: str | None,
    temperature: float | None,
    max_tokens: int | None,
    timeout: int | None,
    json_output: bool,
    chroma_url: str | None,
    chroma_host: str | None,
    chroma_port: int | None,
    chroma_scheme: str | None,
    chroma_path: Path | None,
    input_prompt: str | None,
) -> None:
    """ run grep.

    Parameters
    ----------
    collection : str | None
        Argument value.
    embedding_model : str
        Argument value.
    top_k : int
        Argument value.
    provider_name : str | None
        Argument value.
    model : str | None
        Argument value.
    system : str | None
        Argument value.
    temperature : float | None
        Argument value.
    max_tokens : int | None
        Argument value.
    timeout : int | None
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
    input_prompt : str | None
        Argument value.

    Returns
    -------
    None
        Returned value.
    """
    if top_k <= 0:
        raise ValueError("--top-k must be > 0")
    # end if

    prompt = resolve_prompt(input_prompt)
    prompt_text = prompt.text
    selected_provider = _resolve_provider(provider_name)
    selected_model = _resolve_model(model)
    collection_name = _resolve_collection_name(collection)
    query_embedding = _embed_prompt(embedding_model, prompt_text)

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
        query_result = col.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["metadatas", "documents", "distances"],
        )
    finally:
        if guard is not None:
            guard.close()
    # end try
        # end if

    sources = _collect_sources(query_result)
    if not sources:
        raise ValueError(f"No matching chunks found in collection '{collection_name}'.")
    # end if

    context = _build_context(sources)
    user_prompt = (
        f"Question:\n{prompt_text}\n\nContext:\n{context}\n\n"
        "Answer in the same language as the question. Use the context above and cite sources like [1], [2]. "
        "If the context is insufficient, say it clearly."
    )

    messages: list[ChatMessage] = []
    if system is not None and system.strip():
        messages.append(ChatMessage.system(system.strip()))
    # end if
    messages.append(ChatMessage.user_with_text(user_prompt))

    response = asyncio.run(
        provider_ask(
            selected_provider,
            selected_model,
            messages,
            ChatOptions(
                temperature=temperature,
                max_tokens=max_tokens,
                timeout_secs=timeout,
                retries=0,
                retry_delay_ms=500,
            ),
        )
    )

    if json_output:
        print_json(
            {
                "collection": collection_name,
                "prompt": prompt_text,
                "answer": response.content,
                "sources": sources,
            }
        )
        return
    # end if

    console.print(response.content.rstrip())
    console.print()
    console.print("Sources:")
    for hit in sources:
        source = hit.get("source") or "unknown"
        chunk = f" chunk={int(hit['chunk_index']) + 1}" if hit.get("chunk_index") is not None else ""
        distance = f" distance={float(hit['distance']):.4f}" if hit.get("distance") is not None else ""
        console.print(f"- [{hit['rank']}] {source} (id={hit['id']}{chunk}{distance})")
    # end for
# end def _run_grep


def _embed_prompt(model: str, prompt: str) -> list[float]:
    """ embed prompt.

    Parameters
    ----------
    model : str
        Argument value.
    prompt : str
        Argument value.

    Returns
    -------
    list[float]
        Returned value.
    """
    vectors = embed_chunks_with_provider(EmbeddingProvider.FIREWORKS, model, [prompt])
    if not vectors:
        raise ValueError("Embedding provider returned no vector for prompt.")
    # end if
    return [float(value) for value in vectors[0]]
# end def _embed_prompt


def _collect_sources(query_result: dict[str, Any]) -> list[dict[str, Any]]:
    """ collect sources.

    Parameters
    ----------
    query_result : dict[str, Any]
        Argument value.

    Returns
    -------
    list[dict[str, Any]]
        Returned value.
    """
    ids = (query_result.get("ids") or [[]])[0]
    docs = ((query_result.get("documents") or [[]])[0])
    metadatas = ((query_result.get("metadatas") or [[]])[0])
    distances = ((query_result.get("distances") or [[]])[0])
    hits: list[dict[str, Any]] = []
    for idx, item_id in enumerate(ids):
        metadata = metadatas[idx] if idx < len(metadatas) else None
        source = metadata.get("source") if isinstance(metadata, dict) else None
        chunk_index = metadata.get("chunk_index") if isinstance(metadata, dict) else None
        distance = distances[idx] if idx < len(distances) else None
        document = docs[idx] if idx < len(docs) else ""
        hits.append(
            {
                "rank": idx + 1,
                "id": item_id,
                "source": source,
                "chunk_index": chunk_index,
                "distance": distance,
                "document": document,
            }
        )
    # end for
    return hits
# end def _collect_sources


def _build_context(sources: list[dict[str, Any]]) -> str:
    """ build context.

    Parameters
    ----------
    sources : list[dict[str, Any]]
        Argument value.

    Returns
    -------
    str
        Returned value.
    """
    lines: list[str] = []
    for hit in sources:
        source = hit.get("source") or "unknown"
        distance = f"{float(hit['distance']):.4f}" if hit.get("distance") is not None else "n/a"
        lines.append(
            f"[{hit['rank']}] source={source} id={hit['id']} distance={distance}\n{str(hit.get('document', '')).strip()}"
        )
    # end for
    return "\n\n".join(lines)
# end def _build_context


def _resolve_provider(cli_provider: str | None) -> Provider:
    """ resolve provider.

    Parameters
    ----------
    cli_provider : str | None
        Argument value.

    Returns
    -------
    Provider
        Returned value.
    """
    if cli_provider:
        return Provider(cli_provider)
    # end if
    env_provider = os.getenv("MP_PROVIDER")
    if env_provider:
        value = env_provider.strip().lower()
        if value not in {"openai", "fireworks"}:
            raise ValueError(
                f"Invalid MP_PROVIDER '{value}'. Supported values: openai, fireworks."
            )
        # end if
        return Provider(value)
    # end if
    return Provider.OPENAI
# end def _resolve_provider


def _resolve_model(cli_model: str | None) -> str:
    """ resolve model.

    Parameters
    ----------
    cli_model : str | None
        Argument value.

    Returns
    -------
    str
        Returned value.
    """
    if cli_model and cli_model.strip():
        return cli_model.strip()
    # end if
    env_model = os.getenv("MP_MODEL", "").strip()
    if env_model:
        return env_model
    # end if
    raise ValueError("No model provided. Use --model or set MP_MODEL.")
# end def _resolve_model


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
