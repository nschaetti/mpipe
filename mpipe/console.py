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

"""mpipe.console module.

Notes
-----
This module is part of the Python port of the `mpipe` project.
"""
import json
from typing import Any

from rich.console import Console


console = Console(markup=False)
err_console = Console(stderr=True, markup=False)


def print_json(payload: Any, stderr: bool = False) -> None:
    """Print json.

    Parameters
    ----------
    payload : Any
        Argument value.
    stderr : bool
        Argument value.

    Returns
    -------
    None
        Returned value.
    """
    target = err_console if stderr else console
    target.print(json.dumps(payload, ensure_ascii=False), markup=False)
# end def print_json
