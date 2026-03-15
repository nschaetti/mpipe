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

"""mpipe.rchain.chat_runtime module.

Notes
-----
This module is part of the Python port of the `mpipe` project.
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any

import requests


@dataclass(slots=True)
class RetryConfig:
    """Retryconfig.

    Notes
    -----
    This class follows the same role and hierarchy as the Rust implementation.
    """
    timeout_secs: int | None
    retries: int
    retry_delay_ms: int
# end class RetryConfig


class RequestFailure(Exception):
    """Requestfailure.

    Notes
    -----
    This class follows the same role and hierarchy as the Rust implementation.
    """
    pass
# end class RequestFailure


class RequestFailureRequest(RequestFailure):
    """Requestfailurerequest.

    Notes
    -----
    This class follows the same role and hierarchy as the Rust implementation.
    """
    def __init__(self, source: Exception) -> None:
        """__init__.

        Parameters
        ----------
        source : Exception
            Argument value.

        Returns
        -------
        None
            Returned value.
        """
        super().__init__(str(source))
        self.source = source
    # end def __init__
# end class RequestFailureRequest


class RequestFailureApi(RequestFailure):
    """Requestfailureapi.

    Notes
    -----
    This class follows the same role and hierarchy as the Rust implementation.
    """
    def __init__(self, status_code: int, body: str) -> None:
        """__init__.

        Parameters
        ----------
        status_code : int
            Argument value.
        body : str
            Argument value.

        Returns
        -------
        None
            Returned value.
        """
        super().__init__(f"API error {status_code}: {body}")
        self.status_code = status_code
        self.body = body
    # end def __init__
# end class RequestFailureApi


async def send_chat_request_with_retry(
    client: requests.Session,
    url: str,
    api_key: str,
    payload: dict[str, Any],
    config: RetryConfig,
) -> requests.Response:
    """Send chat request with retry.

    Parameters
    ----------
    client : requests.Session
        Argument value.
    url : str
        Argument value.
    api_key : str
        Argument value.
    payload : dict[str, Any]
        Argument value.
    config : RetryConfig
        Argument value.

    Returns
    -------
    requests.Response
        Returned value.
    """
    max_attempts = max(1, config.retries + 1)
    attempt = 0

    while True:
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        try:
            response = await asyncio.to_thread(
                client.post,
                url,
                headers=headers,
                json=payload,
                timeout=config.timeout_secs,
            )
        except requests.RequestException as err:
            can_retry = _is_retryable_request_error(err) and attempt + 1 < max_attempts
            if can_retry:
                await asyncio.sleep(_retry_delay(attempt, config.retry_delay_ms))
                attempt += 1
                continue
            # end if
            raise RequestFailureRequest(err) from err
        # end try

        if response.ok:
            return response
        # end if

        status_code = response.status_code
        body = response.text
        can_retry = _is_retryable_status(status_code) and attempt + 1 < max_attempts
        if can_retry:
            await asyncio.sleep(_retry_delay(attempt, config.retry_delay_ms))
            attempt += 1
            continue
        # end if

        raise RequestFailureApi(status_code, body)
    # end while
# end def send_chat_request_with_retry


def _is_retryable_status(status: int) -> bool:
    """ is retryable status.

    Parameters
    ----------
    status : int
        Argument value.

    Returns
    -------
    bool
        Returned value.
    """
    return status == 429 or 500 <= status <= 599
# end def _is_retryable_status


def _is_retryable_request_error(err: requests.RequestException) -> bool:
    """ is retryable request error.

    Parameters
    ----------
    err : requests.RequestException
        Argument value.

    Returns
    -------
    bool
        Returned value.
    """
    return isinstance(
        err,
        (
            requests.Timeout,
            requests.ConnectionError,
            requests.ChunkedEncodingError,
        ),
    )
# end def _is_retryable_request_error


def _retry_delay(attempt: int, base_ms: int) -> float:
    """ retry delay.

    Parameters
    ----------
    attempt : int
        Argument value.
    base_ms : int
        Argument value.

    Returns
    -------
    float
        Returned value.
    """
    delay_ms = min(base_ms * (2**attempt), 30_000)
    return delay_ms / 1000.0
# end def _retry_delay
