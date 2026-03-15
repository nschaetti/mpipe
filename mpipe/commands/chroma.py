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

"""mpipe.commands.chroma module.

Notes
-----
This module is part of the Python port of the `mpipe` project.
"""
from __future__ import annotations

import os
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

import chromadb


DEFAULT_CHROMA_LOCAL_HOST = "127.0.0.1"
DEFAULT_CHROMA_PORT = 8000
LOCAL_CHROMA_START_TIMEOUT_SECS = 10.0
LOCAL_CHROMA_POLL_INTERVAL_SECS = 0.25


@dataclass(slots=True)
class ChromaConnectArgs:
    """Chromaconnectargs.

    Notes
    -----
    This class follows the same role and hierarchy as the Rust implementation.
    """
    chroma_url: str | None = None
    chroma_host: str | None = None
    chroma_port: int | None = None
    chroma_scheme: str | None = None
    chroma_path: Path | None = None
# end class ChromaConnectArgs


@dataclass(slots=True)
class LocalChromaGuard:
    """Localchromaguard.

    Notes
    -----
    This class follows the same role and hierarchy as the Rust implementation.
    """
    process: subprocess.Popen[bytes] | None = None

    def close(self) -> None:
        """Close.

        Parameters
        ----------
        None
            This callable does not accept explicit parameters.

        Returns
        -------
        None
            Returned value.
        """
        if self.process is None:
            return
        # end if
        self.process.kill()
        self.process.wait(timeout=3)
        self.process = None
    # end def close
# end class LocalChromaGuard


def connect(args: ChromaConnectArgs) -> tuple[chromadb.api.ClientAPI, LocalChromaGuard | None]:
    """Connect.

    Parameters
    ----------
    args : ChromaConnectArgs
        Argument value.

    Returns
    -------
    tuple[chromadb.api.ClientAPI, LocalChromaGuard | None]
        Returned value.
    """
    connection = _resolve_connection(args)
    guard = _start_local_chroma_if_needed(connection)
    if connection["url"] is not None:
        client = chromadb.HttpClient(host=connection["host"], port=connection["port"])
    else:
        client = chromadb.Client()
    # end if
    return client, guard
# end def connect


def _resolve_connection(args: ChromaConnectArgs) -> dict[str, object | None]:
    """ resolve connection.

    Parameters
    ----------
    args : ChromaConnectArgs
        Argument value.

    Returns
    -------
    dict[str, object | None]
        Returned value.
    """
    chroma_path = args.chroma_path or _env_path("CHROMA_PATH")
    if chroma_path is not None:
        if args.chroma_url:
            raise ValueError("--chroma-url cannot be used with --chroma-path/CHROMA_PATH.")
        # end if
        scheme = args.chroma_scheme or os.getenv("CHROMA_SCHEME", "http")
        if scheme.lower() != "http":
            raise ValueError(f"--chroma-path requires --chroma-scheme=http (got '{scheme}').")
        # end if
        host = args.chroma_host or os.getenv("CHROMA_HOST", DEFAULT_CHROMA_LOCAL_HOST)
        if "://" in host:
            raise ValueError("--chroma-host must be a hostname when using --chroma-path (no scheme).")
        # end if
        port = args.chroma_port or _env_port() or DEFAULT_CHROMA_PORT
        return {
            "url": f"http://{host}:{port}",
            "host": host,
            "port": port,
            "path": chroma_path,
            "local": True,
        }
    # end if

    chroma_url = args.chroma_url
    if chroma_url:
        return {"url": chroma_url, "host": None, "port": None, "path": None, "local": False}
    # end if

    host = args.chroma_host or os.getenv("CHROMA_HOST")
    port = args.chroma_port or _env_port()
    scheme = args.chroma_scheme or os.getenv("CHROMA_SCHEME")
    if host is None and port is None and scheme is None:
        return {"url": None, "host": None, "port": None, "path": None, "local": False}
    # end if

    if host and "://" in host and scheme is None and port is None:
        parsed = host.split("://", 1)[1]
        host_part, _, port_part = parsed.partition(":")
        return {
            "url": host,
            "host": host_part,
            "port": int(port_part) if port_part else DEFAULT_CHROMA_PORT,
            "path": None,
            "local": False,
        }
    # end if

    host = host or "localhost"
    port = port or DEFAULT_CHROMA_PORT
    scheme = scheme or "http"
    return {
        "url": f"{scheme}://{host}:{port}",
        "host": host,
        "port": port,
        "path": None,
        "local": False,
    }
# end def _resolve_connection


def _start_local_chroma_if_needed(connection: dict[str, object | None]) -> LocalChromaGuard | None:
    """ start local chroma if needed.

    Parameters
    ----------
    connection : dict[str, object | None]
        Argument value.

    Returns
    -------
    LocalChromaGuard | None
        Returned value.
    """
    if not connection.get("local"):
        return None
    # end if
    path = connection.get("path")
    host = connection.get("host")
    port = connection.get("port")
    url = connection.get("url")
    if not isinstance(path, Path) or not isinstance(host, str) or not isinstance(port, int):
        return None
    # end if

    path.mkdir(parents=True, exist_ok=True)
    process = subprocess.Popen(
        ["chroma", "run", "--path", str(path), "--host", host, "--port", str(port)],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    _wait_for_chroma_ready(str(url), host, port)
    return LocalChromaGuard(process=process)
# end def _start_local_chroma_if_needed


def _wait_for_chroma_ready(url: str, host: str, port: int) -> None:
    """ wait for chroma ready.

    Parameters
    ----------
    url : str
        Argument value.
    host : str
        Argument value.
    port : int
        Argument value.

    Returns
    -------
    None
        Returned value.
    """
    deadline = time.monotonic() + LOCAL_CHROMA_START_TIMEOUT_SECS
    last_error = "unknown"
    while time.monotonic() < deadline:
        try:
            client = chromadb.HttpClient(host=host, port=port)
            client.heartbeat()
            return
        except Exception as err:
            last_error = str(err)
            time.sleep(LOCAL_CHROMA_POLL_INTERVAL_SECS)
        # end try
    # end while
    raise ValueError(
        f"Local ChromaDB did not become ready at {url} within {LOCAL_CHROMA_START_TIMEOUT_SECS:.0f}s ({last_error})."
    )
# end def _wait_for_chroma_ready


def _env_port() -> int | None:
    """ env port.

    Parameters
    ----------
    None
        This callable does not accept explicit parameters.

    Returns
    -------
    int | None
        Returned value.
    """
    raw = os.getenv("CHROMA_PORT")
    if raw is None:
        return None
    # end if
    trimmed = raw.strip()
    if trimmed == "":
        return None
    # end if
    try:
        return int(trimmed)
    except ValueError as err:
        raise ValueError(f"Invalid CHROMA_PORT '{trimmed}': expected integer 0-65535") from err
    # end try
# end def _env_port


def _env_path(key: str) -> Path | None:
    """ env path.

    Parameters
    ----------
    key : str
        Argument value.

    Returns
    -------
    Path | None
        Returned value.
    """
    value = os.getenv(key, "").strip()
    return Path(value) if value else None
# end def _env_path
