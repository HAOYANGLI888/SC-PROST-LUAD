"""Project path helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


STANDARD_DIRS = [
    "configs",
    "src",
    "scripts",
    "notebooks",
    "data/raw",
    "data/processed",
    "data/metadata",
    "outputs",
    "outputs/figures",
    "outputs/tables",
    "outputs/checkpoints",
    "outputs/logs",
    "outputs/reports",
    "outputs/audit",
    "docs",
    "tests",
]


REQUIRED_STAGE0_FILES = [
    "README.md",
    "AGENTS.md",
    "environment.yml",
    "requirements.txt",
    "configs/base.yaml",
    "configs/data.yaml",
    "configs/model.yaml",
    "docs/data_directory_spec.md",
    "docs/manual_download_guide.md",
]


@dataclass(frozen=True)
class ProjectPaths:
    """Resolved project paths used by scripts and tests."""

    root: Path

    @classmethod
    def from_root(cls, root: str | Path) -> "ProjectPaths":
        return cls(Path(root).resolve())

    def path(self, relative_path: str | Path) -> Path:
        return self.root / Path(relative_path)

    @property
    def data_raw(self) -> Path:
        return self.path("data/raw")

    @property
    def data_processed(self) -> Path:
        return self.path("data/processed")

    @property
    def data_metadata(self) -> Path:
        return self.path("data/metadata")

    @property
    def outputs(self) -> Path:
        return self.path("outputs")

    @property
    def figures(self) -> Path:
        return self.path("outputs/figures")

    @property
    def tables(self) -> Path:
        return self.path("outputs/tables")

    @property
    def checkpoints(self) -> Path:
        return self.path("outputs/checkpoints")

    @property
    def logs(self) -> Path:
        return self.path("outputs/logs")

    @property
    def reports(self) -> Path:
        return self.path("outputs/reports")

    @property
    def audit(self) -> Path:
        return self.path("outputs/audit")

    def ensure_standard_dirs(self) -> None:
        for relative_dir in STANDARD_DIRS:
            self.path(relative_dir).mkdir(parents=True, exist_ok=True)


def find_project_root(start: str | Path | None = None) -> Path:
    """Find the project root by walking upward from *start*."""

    current = Path(start or Path.cwd()).resolve()
    if current.is_file():
        current = current.parent

    for candidate in [current, *current.parents]:
        if (candidate / "configs" / "base.yaml").exists() and (
            candidate / "src" / "scprost_luad" / "__init__.py"
        ).exists():
            return candidate

    raise FileNotFoundError(
        f"Could not find SC-PROST-LUAD root from {current}. "
        "Run commands from the project directory or pass --root."
    )


def relative_to_root(path: Path, root: Path) -> str:
    """Return a stable POSIX-style path relative to the project root."""

    return path.resolve().relative_to(root.resolve()).as_posix()
