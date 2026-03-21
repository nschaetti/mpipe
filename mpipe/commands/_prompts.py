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
"""Prompt templates for ``mpipe tools create``.

Attributes:
    TOOL_COMMAND_EXTRACT: Base prompt used to convert CLI help text into a
        strict JSON tool bundle.
    TOOL_COMMAND_CREATE_ERROR_JSON: Repair prompt used when model output is not
        valid JSON.
    TOOL_COMMAND_CREATE_MISSING_TOOL: Repair prompt used when the top-level
        ``tool`` key is missing.
    TOOL_COMMAND_CREATE_MISSING_CLI_MAP: Repair prompt used when the top-level
        ``cli_map`` key is missing.
"""

TOOL_COMMAND_EXTRACT = """You are a CLI help-to-tool parser.

Your task is to convert the raw output of a command's `--help` into a single STRICT JSON object with exactly two top-level keys:

- "tool": a function-calling tool definition
- "cli_map": a mapping used to reconstruct the CLI command

Hard requirements:
- Output JSON only.
- No markdown.
- No explanations.
- No text before or after the JSON.
- Use strict valid JSON, not Python dict syntax.
- Use double quotes for all keys and strings.
- Use true, false, null in JSON form.
- Never invent arguments, options, enums, default values, or subcommands not clearly supported by the help text.
- If something is unclear, use the most conservative interpretation.
- If a type is unclear, use "string".

You must return exactly one JSON object with this exact top-level structure:
{
  "tool": {
    "type": "function",
    "function": {
      "name": "tool_name_in_snake_case",
      "description": "short faithful description",
      "strict": true,
      "parameters": {
        "type": "object",
        "properties": {},
        "required": [],
        "additionalProperties": false
      }
    }
  },
  "cli_map": {}
}
No other top-level keys are allowed.

The output must contain exactly two top-level keys: "tool" and "cli_map".

"cli_map" must be a top-level sibling of "tool", not nested inside "tool" and not nested inside "tool.function".

Invalid:
{
  "tool": {
    ...,
    "cli_map": {}
  }
}

Valid:
{
  "tool": {},
  "cli_map": {}
}

Important parsing rules:
1. The "Usage:" line is the primary source of truth for positional arguments.
2. The "Arguments:" section is secondary and may refine positional arguments.
3. If "Arguments:" is empty or incomplete, still extract positional arguments from "Usage:".
4. Tokens like [OPTION], [OPTIONS], [OPTION]... are structural markers and must NOT become parameters.
5. Real placeholders like FILE, PATH, DIR, PATTERN, COMMAND, TARGET, QUERY, TEXT, etc. that appear in "Usage:" must become parameters.
6. If a positional appears as [FILE], it is optional.
7. If a positional appears as FILE, it is required.
8. If a positional appears as [FILE]... or FILE..., it is repeatable and must become an array.
9. Do not drop positional arguments just because their descriptions are missing.

Rules for "tool":
- "tool" must be suitable for function/tool calling.
- Use readable JSON parameter names in snake_case.
- Prefer long option names for parameter names.
- If only a short option exists, derive a readable snake_case name from its description.
- Positional arguments should also get readable snake_case names.
- Flags become boolean.
- Options with values become typed parameters.
- Repeatable arguments/options become arrays.
- Only include "enum" if the help explicitly gives a closed set of allowed values for that exact parameter.
- Ignore --help and --version.

Rules for "cli_map":
For every parameter present in tool.function.parameters.properties, there must be exactly one entry in "cli_map" with the same key.

Each cli_map entry must have:
- "kind": one of "positional", "flag", "option"

For positional arguments, use:
{
  "kind": "positional",
  "position": 0,
  "placeholder": "FILE",
  "repeatable": true
}

For boolean flags, use:
{
  "kind": "flag",
  "short": "-l",
  "long": "--long"
}

For options with values, use:
{
  "kind": "option",
  "short": "-T",
  "long": "--tabsize",
  "placeholder": "COLS",
  "repeatable": false,
  "value_mode": "separate"
}

Additional cli_map rules:
- "position" is zero-based.
- "placeholder" should preserve the original CLI placeholder when available, like FILE, PATH, COLS.
- "repeatable" is required for positional and option entries when applicable.
- "value_mode" should be:
  - "equals" for forms like --color=<when>
  - "separate" for forms like --ignore PATTERN or -T COLS
  - "either" only if the help clearly supports both
- Use null for missing short or long forms if needed.
- Do not invent aliases.

Type rules:
- boolean for flags
- integer for explicit integer placeholders like INT, N, NUM, COUNT, PORT, COLS
- number for explicit decimal/numeric placeholders like FLOAT, DOUBLE, RATIO, SECONDS
- string otherwise
- array for repeatable values

Example expectation:
If usage is:
Usage: ls [OPTION]... [FILE]...

Then:
- [OPTION]... does not become a parameter
- [FILE]... becomes an optional array parameter such as "files"
- cli_map["files"] must be:
  {
    "kind": "positional",
    "position": 0,
    "placeholder": "FILE",
    "repeatable": true
  }

Now parse the following help text and output JSON only:

{{HELP_TEXT}}
"""


TOOL_COMMAND_CREATE_ERROR_JSON = """Your previous response was invalid because it was not valid JSON.

Parser error:
{{PARSER_ERROR}}

Fix your previous answer and return it again.

Requirements:
- Return JSON only.
- No markdown.
- No explanation.
- No text before or after the JSON.
- The output must be parseable by Python's json.loads().
- Use double quotes for keys and strings.
- Use JSON literals: true, false, null.
- Keep the same intended content, but correct the formatting and structure.

Remember: the output must contain exactly two top-level keys:
- "tool"
- "cli_map"
"""


TOOL_COMMAND_CREATE_MISSING_TOOL = """Your previous response was invalid because the top-level key "tool" is missing.

Fix your previous answer and return it again.

Requirements:
- Return JSON only.
- No markdown.
- No explanation.
- No text before or after the JSON.
- The output must contain exactly two top-level keys:
  - "tool"
  - "cli_map"
- "tool" must be a top-level object, not nested inside another key.
- "cli_map" must remain a top-level sibling of "tool".

Expected top-level structure:
{
  "tool": {
    "type": "function",
    "function": {
      "name": "...",
      "description": "...",
      "strict": true,
      "parameters": {
        "type": "object",
        "properties": {},
        "required": [],
        "additionalProperties": false
      }
    }
  },
  "cli_map": {}
}
"""


TOOL_COMMAND_CREATE_MISSING_CLI_MAP = """Your previous response was invalid because the top-level key "cli_map" is missing.

Fix your previous answer and return it again.

Requirements:
- Return JSON only.
- No markdown.
- No explanation.
- No text before or after the JSON.
- The output must contain exactly two top-level keys:
  - "tool"
  - "cli_map"
- "cli_map" must be a top-level object, not nested inside "tool" and not nested inside "tool.function".
- "tool" and "cli_map" must be siblings at the top level.

Invalid example:
{
  "tool": {
    "...": "...",
    "cli_map": {}
  }
}

Valid shape:
{
  "tool": { ... },
  "cli_map": { ... }
}
"""
