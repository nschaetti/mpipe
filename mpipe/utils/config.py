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

"""Utilities for creating configuration directories used by mpipe."""

from pathlib import Path


__all__ = ["create_config"]


def create_config(
        path: Path = Path(".config/mpipe"),
        subdir: str | None = None,
) -> Path:
    """Create the base configuration directory and an optional subdirectory.

    Args:
        path: Base configuration path to create.
        subdir: Optional subdirectory name to create inside ``path``.

    Returns:
        The created base directory, or the created subdirectory path when
        ``subdir`` is provided.
    """
    path.mkdir(parents=True, exist_ok=True)
    if subdir:
        (path / subdir).mkdir(parents=True, exist_ok=True)
        return path / subdir
    # end if
    return path
# end def create_config
