"""
Compatibility entrypoint for the former dossier record inspection command.

# Diagram omitted - this is a thin CLI wrapper with no significant information flow.
"""

from __future__ import annotations

from inspect_structured_records import main


if __name__ == "__main__":
    raise SystemExit(main())
