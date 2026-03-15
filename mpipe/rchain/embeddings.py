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

"""mpipe.rchain.embeddings module.

Notes
-----
This module is part of the Python port of the `mpipe` project.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass
from enum import Enum

import requests


class EmbeddingProvider(str, Enum):
    """Embeddingprovider.

    Notes
    -----
    This class follows the same role and hierarchy as the Rust implementation.
    """
    OPENAI = "openai"
    FIREWORKS = "fireworks"

    def as_str(self) -> str:
        """As str.

        Parameters
        ----------
        None
            This callable does not accept explicit parameters.

        Returns
        -------
        str
            Returned value.
        """
        return self.value
    # end def as_str
# end class EmbeddingProvider


class ChunkStrategy(str, Enum):
    """Chunkstrategy.

    Notes
    -----
    This class follows the same role and hierarchy as the Rust implementation.
    """
    PARAGRAPH = "paragraph"
    SENTENCE = "sentence"
    TOKEN = "token"

    @classmethod
    def from_str(cls, value: str) -> "ChunkStrategy | None":
        """From str.

        Parameters
        ----------
        value : str
            Argument value.

        Returns
        -------
        'ChunkStrategy | None'
            Returned value.
        """
        raw = value.strip().lower()
        for strategy in cls:
            if strategy.value == raw:
                return strategy
            # end if
        # end for
        return None
    # end def from_str
# end class ChunkStrategy


@dataclass(slots=True)
class EmbeddingsConfig:
    """Embeddingsconfig.

    Notes
    -----
    This class follows the same role and hierarchy as the Rust implementation.
    """
    provider: EmbeddingProvider = EmbeddingProvider.FIREWORKS
    model: str = "accounts/fireworks/models/qwen3-embedding-8b"
    chunk_size: int = 8000
    chunk_overlap: int = 10
    chunk_strategy: ChunkStrategy = ChunkStrategy.PARAGRAPH

    def api_key(self) -> str:
        """Api key.

        Parameters
        ----------
        None
            This callable does not accept explicit parameters.

        Returns
        -------
        str
            Returned value.
        """
        env_key = "OPENAI_API_KEY" if self.provider == EmbeddingProvider.OPENAI else "FIREWORKS_API_KEY"
        value = os.getenv(env_key, "")
        if not value:
            raise ValueError(f"{env_key} is not set in the environment")
        # end if
        return value
    # end def api_key
# end class EmbeddingsConfig


@dataclass(slots=True)
class EmbeddingResult:
    """Embeddingresult.

    Notes
    -----
    This class follows the same role and hierarchy as the Rust implementation.
    """
    chunks: list[str]
    embeddings: list[list[float]]
    model: str
    provider: str
# end class EmbeddingResult


def chunk_text(text: str, strategy: ChunkStrategy, chunk_size: int, overlap_percent: int) -> list[str]:
    """Chunk text.

    Parameters
    ----------
    text : str
        Argument value.
    strategy : ChunkStrategy
        Argument value.
    chunk_size : int
        Argument value.
    overlap_percent : int
        Argument value.

    Returns
    -------
    list[str]
        Returned value.
    """
    if strategy == ChunkStrategy.PARAGRAPH:
        return _chunk_by_paragraph(text, chunk_size, overlap_percent)
    # end if
    if strategy == ChunkStrategy.SENTENCE:
        return _chunk_by_sentence(text, chunk_size, overlap_percent)
    # end if
    return _chunk_by_token(text, chunk_size, overlap_percent)
# end def chunk_text


def embed_texts(config: EmbeddingsConfig, texts: list[str]) -> EmbeddingResult:
    """Embed texts.

    Parameters
    ----------
    config : EmbeddingsConfig
        Argument value.
    texts : list[str]
        Argument value.

    Returns
    -------
    EmbeddingResult
        Returned value.
    """
    all_chunks: list[str] = []
    for text in texts:
        all_chunks.extend(
            chunk_text(text, config.chunk_strategy, config.chunk_size, config.chunk_overlap)
        )
    # end for

    if not all_chunks:
        return EmbeddingResult(
            chunks=[],
            embeddings=[],
            model=config.model,
            provider=config.provider.as_str(),
        )
    # end if

    embeddings = embed_chunks(config, all_chunks)
    return EmbeddingResult(
        chunks=all_chunks,
        embeddings=embeddings,
        model=config.model,
        provider=config.provider.as_str(),
    )
# end def embed_texts


def embed_chunks(config: EmbeddingsConfig, chunks: list[str]) -> list[list[float]]:
    """Embed chunks.

    Parameters
    ----------
    config : EmbeddingsConfig
        Argument value.
    chunks : list[str]
        Argument value.

    Returns
    -------
    list[list[float]]
        Returned value.
    """
    if config.provider == EmbeddingProvider.OPENAI:
        return _embed_chunks_openai(config.model, config.api_key(), chunks)
    # end if
    return _embed_chunks_fireworks(config.model, config.api_key(), chunks)
# end def embed_chunks


def embed_chunks_with_provider(
    provider: EmbeddingProvider,
    model: str,
    chunks: list[str],
) -> list[list[float]]:
    """Embed chunks with provider.

    Parameters
    ----------
    provider : EmbeddingProvider
        Argument value.
    model : str
        Argument value.
    chunks : list[str]
        Argument value.

    Returns
    -------
    list[list[float]]
        Returned value.
    """
    config = EmbeddingsConfig(provider=provider, model=model)
    return embed_chunks(config, chunks)
# end def embed_chunks_with_provider


def _chunk_by_paragraph(text: str, chunk_size: int, overlap_percent: int) -> list[str]:
    """ chunk by paragraph.

    Parameters
    ----------
    text : str
        Argument value.
    chunk_size : int
        Argument value.
    overlap_percent : int
        Argument value.

    Returns
    -------
    list[str]
        Returned value.
    """
    del overlap_percent
    paragraphs = [p for p in text.split("\n\n") if p.strip()]
    chunks: list[str] = []
    current: list[str] = []
    current_size = 0
    for paragraph in paragraphs:
        paragraph_size = len(paragraph)
        if current and current_size + paragraph_size > chunk_size:
            chunks.append("\n\n".join(current).strip())
            current = []
            current_size = 0
        # end if
        current.append(paragraph)
        current_size += paragraph_size + (2 if current_size > 0 else 0)
    # end for
    if current:
        chunks.append("\n\n".join(current).strip())
    # end if
    if not chunks and text.strip():
        chunks.append(text.strip())
    # end if
    return chunks
# end def _chunk_by_paragraph


def _chunk_by_sentence(text: str, chunk_size: int, overlap_percent: int) -> list[str]:
    """ chunk by sentence.

    Parameters
    ----------
    text : str
        Argument value.
    chunk_size : int
        Argument value.
    overlap_percent : int
        Argument value.

    Returns
    -------
    list[str]
        Returned value.
    """
    del overlap_percent
    sentences = [s.strip() for s in re.split(r"(?<=[.!?¿¡])\s+", text) if s.strip()]
    chunks: list[str] = []
    current: list[str] = []
    current_size = 0
    for sentence in sentences:
        sentence_size = len(sentence)
        if current and current_size + sentence_size > chunk_size:
            chunks.append(" ".join(current).strip())
            current = []
            current_size = 0
        # end if
        current.append(sentence)
        current_size += sentence_size + (1 if current_size > 0 else 0)
    # end for
    if current:
        chunks.append(" ".join(current).strip())
    # end if
    if not chunks and text.strip():
        chunks.append(text.strip())
    # end if
    return chunks
# end def _chunk_by_sentence


def _chunk_by_token(text: str, chunk_size: int, overlap_percent: int) -> list[str]:
    """ chunk by token.

    Parameters
    ----------
    text : str
        Argument value.
    chunk_size : int
        Argument value.
    overlap_percent : int
        Argument value.

    Returns
    -------
    list[str]
        Returned value.
    """
    words = text.split()
    chunks: list[str] = []
    current: list[str] = []
    current_size = 0
    for word in words:
        word_size = max(1, len(re.sub(r"\W+", "", word)))
        if current and current_size + word_size > chunk_size:
            chunks.append(" ".join(current))
            overlap_words = (chunk_size * overlap_percent) // 100 // 5
            current = current[max(0, len(current) - min(overlap_words, len(current))) :]
            current_size = sum(len(item) for item in current)
        # end if
        current.append(word)
        current_size += word_size
    # end for
    if current:
        chunks.append(" ".join(current))
    # end if
    if not chunks and text.strip():
        chunks.append(text.strip())
    # end if
    return chunks
# end def _chunk_by_token


def _embed_chunks_fireworks(model: str, api_key: str, chunks: list[str]) -> list[list[float]]:
    """ embed chunks fireworks.

    Parameters
    ----------
    model : str
        Argument value.
    api_key : str
        Argument value.
    chunks : list[str]
        Argument value.

    Returns
    -------
    list[list[float]]
        Returned value.
    """
    url = "https://api.fireworks.ai/inference/v1/embeddings"
    session = requests.Session()
    vectors: list[list[float]] = []
    for chunk in chunks:
        response = session.post(
            url,
            headers={"Authorization": f"Bearer {api_key}"},
            json={"model": model, "input": chunk},
            timeout=60,
        )
        if not response.ok:
            raise RuntimeError(f"Fireworks API error {response.status_code}: {response.text}")
        # end if
        body = response.json()
        data = body.get("data")
        if not isinstance(data, list) or not data:
            raise RuntimeError("Missing embedding data from Fireworks API")
        # end if
        embedding = data[0].get("embedding")
        if not isinstance(embedding, list):
            raise RuntimeError("Missing embedding data from Fireworks API")
        # end if
        vectors.append([float(value) for value in embedding])
    # end for
    return vectors
# end def _embed_chunks_fireworks


def _embed_chunks_openai(model: str, api_key: str, chunks: list[str]) -> list[list[float]]:
    """ embed chunks openai.

    Parameters
    ----------
    model : str
        Argument value.
    api_key : str
        Argument value.
    chunks : list[str]
        Argument value.

    Returns
    -------
    list[list[float]]
        Returned value.
    """
    url = "https://api.openai.com/v1/embeddings"
    response = requests.post(
        url,
        headers={"Authorization": f"Bearer {api_key}"},
        json={"model": model, "input": chunks},
        timeout=60,
    )
    if not response.ok:
        raise RuntimeError(f"OpenAI API error {response.status_code}: {response.text}")
    # end if
    body = response.json()
    data = body.get("data")
    if not isinstance(data, list):
        raise RuntimeError("Missing embedding data from OpenAI API")
    # end if
    vectors: list[list[float]] = []
    for item in data:
        embedding = item.get("embedding") if isinstance(item, dict) else None
        if not isinstance(embedding, list):
            raise RuntimeError("Missing embedding array")
        # end if
        vectors.append([float(value) for value in embedding])
    # end for
    vectors.sort(key=len)
    return vectors
# end def _embed_chunks_openai
