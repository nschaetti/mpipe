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

"""Data models for tool-calling payloads and CLI argument mappings."""
from __future__ import annotations

import base64
import io
import json
from dataclasses import dataclass
from typing import Any, Literal, Optional, Union

from PIL import Image
from pydantic import BaseModel


class ToolFunctionParameters(BaseModel):
    """JSON Schema object describing a tool function parameters object.

    Attributes:
        type: JSON schema type, always ``"object"``.
        properties: Parameter schema map keyed by parameter name.
        required: Names of required parameters.
        additionalProperties: Whether unknown properties are accepted.
    """

    type: Literal["object"] = "object"
    properties: dict[str, Any]
    required: list[str]
    additionalProperties: bool
# end class ToolFunctionParameters


class ToolFunction(BaseModel):
    """Function metadata exposed to model tool-calling APIs.

    Attributes:
        name: Tool function name.
        description: Human-readable function description.
        strict: Whether strict schema adherence is requested.
        parameters: JSON schema definition for function parameters.
    """

    name: str
    description: str
    strict: bool
    parameters: ToolFunctionParameters
# end class ToolFunction


class ToolDef(BaseModel):
    """Top-level tool wrapper used by chat completion APIs.

    Attributes:
        type: Tool kind, always ``"function"``.
        function: Function metadata payload.
    """

    type: Literal["function"] = "function"
    function: ToolFunction

    def to_json(self) -> dict[str, Any]:
        """Serialize this model to a JSON-compatible dictionary.

        Returns:
            A dictionary suitable for JSON encoding and API transport.
        """
        return self.model_dump(mode="json")
    # end def to_json
# end class ToolDef


class CliMapPositional(BaseModel):
    """CLI positional argument mapping for one tool parameter.

    Attributes:
        kind: Mapping type discriminator, always ``"positional"``.
        position: Zero-based positional index in CLI order.
        placeholder: Original CLI placeholder token.
        repeatable: Whether this positional argument is repeatable.
    """

    kind: Literal["positional"] = "positional"
    position: int
    placeholder: str
    repeatable: bool
# end class CliMapPositional


class CliMapFlag(BaseModel):
    """CLI boolean flag mapping for one tool parameter.

    Attributes:
        kind: Mapping type discriminator, always ``"flag"``.
        short: Optional short flag form (for example ``-v``).
        long: Optional long flag form (for example ``--verbose``).
    """

    kind: Literal["flag"] = "flag"
    short: Optional[str] = None
    long: Optional[str] = None
# end class CliMapFlag


class CliMapOption(BaseModel):
    """CLI option-with-value mapping for one tool parameter.

    Attributes:
        kind: Mapping type discriminator, always ``"option"``.
        short: Optional short option form.
        long: Optional long option form.
        placeholder: Original value placeholder token.
        repeatable: Whether the option can be provided multiple times.
        value_mode: How values are passed (equals, separate, or either).
    """

    kind: Literal["option"] = "option"
    short: Optional[str] = None
    long: Optional[str] = None
    placeholder: str
    repeatable: bool
    value_mode: Literal["equals", "separate", "either"]
# end class CliMapOption


CliMapEntry = Union[CliMapPositional, CliMapFlag, CliMapOption]


class ToolBundle(BaseModel):
    """Bundle containing both tool schema and CLI reconstruction metadata.

    Attributes:
        tool: Tool definition payload.
        cli_map: Mapping from tool parameter names to CLI argument metadata.
    """

    tool: ToolDef
    cli_map: dict[str, CliMapEntry]

    def to_json(self) -> dict[str, Any]:
        """Serialize this model to a JSON-compatible dictionary.

        Returns:
            A dictionary suitable for JSON encoding and file storage.
        """
        return self.model_dump(mode="json")
    # end def to_json

    @classmethod
    def json_schema(cls) -> dict[str, Any]:
        """Return the JSON Schema for the bundle model.

        Returns:
            The Pydantic-generated JSON Schema dictionary.
        """
        return cls.model_json_schema()
    # end def json_schema
# end class ToolBundle


class ToolDefinition(ToolDef):
    """Backward-compatible alias around ``ToolDef`` with legacy constructors."""

    @classmethod
    def from_function(cls, function: ToolFunction) -> "ToolDefinition":
        """Create a tool definition from a function payload.

        Args:
            function: Function metadata to wrap.

        Returns:
            A ``ToolDefinition`` instance wrapping ``function``.
        """
        return cls(function=function)
    # end def from_function
# end class ToolDefinition


@dataclass(slots=True)
class ToolCall:
    """A tool call emitted by a model response.

    Attributes:
        id: Provider-supplied tool call identifier.
        name: Tool/function name requested by the model.
        args: Parsed or raw argument payload for the call.
    """
    id: str
    name: str
    args: Any

    def to_json(self) -> dict[str, Any]:
        """Serialize the tool call to provider-compatible JSON payload.

        Returns:
            A dictionary matching OpenAI-style tool call structure.
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
    """Convert arbitrary image bytes to PNG and return base64 text.

    Args:
        data: Raw input image bytes.

    Returns:
        Base64-encoded PNG bytes represented as an ASCII string.
    """
    image = Image.open(io.BytesIO(data))
    output = io.BytesIO()
    image.save(output, format="PNG")
    return base64.b64encode(output.getvalue()).decode("ascii")
# end def encode_image_base64_from_bytes
