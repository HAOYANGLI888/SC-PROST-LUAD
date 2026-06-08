"""Audit report writer used by every stage."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Iterable


def _as_list(items: Iterable[str]) -> list[str]:
    return [str(item) for item in items]


def write_audit_report(
    output_path: str | Path,
    stage: str,
    completed: Iterable[str],
    input_files: Iterable[str],
    output_files: Iterable[str],
    commands: Iterable[str],
    potential_issues: Iterable[str],
    next_steps: Iterable[str],
    metadata: dict[str, object] | None = None,
) -> Path:
    """Write a stage audit report in Markdown."""

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    metadata = metadata or {}

    lines: list[str] = [
        f"# {stage} Audit Report",
        "",
        f"- Generated: {datetime.now().isoformat(timespec='seconds')}",
        "",
        "## Completed Content",
    ]
    lines.extend(f"- {item}" for item in _as_list(completed))

    lines.extend(["", "## Input Files"])
    lines.extend(f"- `{item}`" for item in _as_list(input_files))

    lines.extend(["", "## Output Files"])
    lines.extend(f"- `{item}`" for item in _as_list(output_files))

    lines.extend(["", "## Run Commands"])
    lines.extend(f"- `{item}`" for item in _as_list(commands))

    lines.extend(["", "## Potential Issues"])
    issue_list = _as_list(potential_issues)
    if issue_list:
        lines.extend(f"- {item}" for item in issue_list)
    else:
        lines.append("- None recorded.")

    lines.extend(["", "## Next Step Suggestions"])
    lines.extend(f"- {item}" for item in _as_list(next_steps))

    if metadata:
        lines.extend(["", "## Metadata"])
        for key, value in metadata.items():
            lines.append(f"- `{key}`: {value}")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path
