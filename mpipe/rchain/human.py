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

"""mpipe.rchain.human module.

Notes
-----
This module is part of the Python port of the `mpipe` project.
"""
from dataclasses import dataclass
from typing import Any


@dataclass(slots=True)
class HumanMessage:
    """Humanmessage.

    Notes
    -----
    This class follows the same role and hierarchy as the Rust implementation.
    """
    content: str | list[dict[str, Any]]

    @classmethod
    def new(cls, content: str) -> "HumanMessage":
        """New.

        Parameters
        ----------
        content : str
            Argument value.

        Returns
        -------
        'HumanMessage'
            Returned value.
        """
        return cls(content=content)
    # end def new

    @classmethod
    def from_parts(cls, parts: list[dict[str, Any]]) -> "HumanMessage":
        """From parts.

        Parameters
        ----------
        parts : list[dict[str, Any]]
            Argument value.

        Returns
        -------
        'HumanMessage'
            Returned value.
        """
        return cls(content=parts)
    # end def from_parts

    def to_json(self) -> str | list[dict[str, Any]]:
        """To json.

        Parameters
        ----------
        None
            This callable does not accept explicit parameters.

        Returns
        -------
        str | list[dict[str, Any]]
            Returned value.
        """
        return self.content
    # end def to_json
# end class HumanMessage
