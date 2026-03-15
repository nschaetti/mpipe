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

"""mpipe.rchain.tools module.

Notes
-----
This module is part of the Python port of the `mpipe` project.
"""
from __future__ import annotations

import base64
import io
import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from PIL import Image


class ToolParamType(str, Enum):
    """Toolparamtype.

    Notes
    -----
    This class follows the same role and hierarchy as the Rust implementation.
    """
    INTEGER = "integer"
    NUMBER = "number"
    STRING = "string"
    BOOLEAN = "boolean"
    OBJECT = "object"
    ARRAY = "array"
# end class ToolParamType


@dataclass(slots=True)
class ToolParam:
    """Toolparam.

    Notes
    -----
    This class follows the same role and hierarchy as the Rust implementation.
    """
    name: str
    kind: ToolParamType
    required: bool
    description: str | None = None
# end class ToolParam


@dataclass(slots=True)
class ToolFunction:
    """Toolfunction.

    Notes
    -----
    This class follows the same role and hierarchy as the Rust implementation.
    """
    name: str
    description: str
    params: list[ToolParam] = field(default_factory=list)

    def with_param(self, param: ToolParam) -> "ToolFunction":
        """With param.

        Parameters
        ----------
        param : ToolParam
            Argument value.

        Returns
        -------
        'ToolFunction'
            Returned value.
        """
        self.params.append(param)
        return self
    # end def with_param

    def to_schema(self) -> dict[str, Any]:
        """To schema.

        Parameters
        ----------
        None
            This callable does not accept explicit parameters.

        Returns
        -------
        dict[str, Any]
            Returned value.
        """
        properties: dict[str, dict[str, str]] = {}
        required: list[str] = []
        for param in self.params:
            param_def: dict[str, str] = {"type": param.kind.value}
            if param.description is not None:
                param_def["description"] = param.description
            # end if
            properties[param.name] = param_def
            if param.required:
                required.append(param.name)
            # end if
        # end for
        schema: dict[str, Any] = {"type": "object", "properties": properties}
        if required:
            schema["required"] = required
        # end if
        return schema
    # end def to_schema
# end class ToolFunction


@dataclass(slots=True)
class ToolDefinition:
    """Tooldefinition.

    Notes
    -----
    This class follows the same role and hierarchy as the Rust implementation.
    """
    function: ToolFunction

    @classmethod
    def from_function(cls, function: ToolFunction) -> "ToolDefinition":
        """From function.

        Parameters
        ----------
        function : ToolFunction
            Argument value.

        Returns
        -------
        'ToolDefinition'
            Returned value.
        """
        return cls(function=function)
    # end def from_function

    def to_json(self) -> dict[str, Any]:
        """To json.

        Parameters
        ----------
        None
            This callable does not accept explicit parameters.

        Returns
        -------
        dict[str, Any]
            Returned value.
        """
        return {
            "type": "function",
            "function": {
                "name": self.function.name,
                "description": self.function.description,
                "parameters": self.function.to_schema(),
            },
        }
    # end def to_json
# end class ToolDefinition


@dataclass(slots=True)
class ToolCall:
    """Toolcall.

    Notes
    -----
    This class follows the same role and hierarchy as the Rust implementation.
    """
    id: str
    name: str
    args: Any

    def to_json(self) -> dict[str, Any]:
        """To json.

        Parameters
        ----------
        None
            This callable does not accept explicit parameters.

        Returns
        -------
        dict[str, Any]
            Returned value.
        """
        if isinstance(self.args, str):
            args_as_string = self.args
        else:
            args_as_string = json.dumps(self.args)
        # end if
        return {
            "id": self.id,
            "type": "function",
            "function": {"name": self.name, "arguments": args_as_string},
        }
    # end def to_json
# end class ToolCall


def encode_image_base64_from_bytes(data: bytes) -> str:
    """Encode image base64 from bytes.

    Parameters
    ----------
    data : bytes
        Argument value.

    Returns
    -------
    str
        Returned value.
    """
    image = Image.open(io.BytesIO(data))
    output = io.BytesIO()
    image.save(output, format="PNG")
    return base64.b64encode(output.getvalue()).decode("ascii")
# end def encode_image_base64_from_bytes
