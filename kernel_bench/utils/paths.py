"""
Path management for kernel benchmarks.

This module provides centralized path configuration for all benchmark artifacts,
replacing scattered hard-coded paths throughout the codebase.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import os


@dataclass
class PathConfig:
    """
    Centralized path configuration for kernel benchmarks.

    This class manages all paths used during benchmarking, including:
    - Compiled artifacts (MLIR, VMFB)
    - Results (CSV, JSON)
    - Tuning outputs
    - Debug dumps

    Attributes:
        workspace: Root workspace directory (typically project root)
        results: Base directory for all benchmark results
        kernels: Base directory for compiled kernel artifacts
        dumps: Optional directory for debug dumps (MLIR IR, etc.)
    """

    workspace: Path
    results: Path
    kernels: Path
    dumps: Optional[Path] = None

    def __post_init__(self):
        """Ensure all paths are Path objects."""
        self.workspace = Path(self.workspace)
        self.results = Path(self.results)
        self.kernels = Path(self.kernels)
        if self.dumps is not None:
            self.dumps = Path(self.dumps)

    def _ensure_dir(self, path: Path) -> Path:
        """
        Ensure a directory exists, creating it if necessary.

        Args:
            path: Directory path to ensure

        Returns:
            The same path (for chaining)
        """
        path.mkdir(parents=True, exist_ok=True)
        return path

    # Result output paths

    def csv_for(self, kernel_type: str) -> Path:
        """
        Get CSV output directory for a kernel type.
        Creates the directory if it doesn't exist.

        Args:
            kernel_type: Kernel type (e.g., "gemm", "attention")

        Returns:
            Path to CSV output directory (guaranteed to exist)
        """
        path = self.results / "csv" / kernel_type
        return self._ensure_dir(path)

    def json_for(self, kernel_type: str) -> Path:
        """
        Get JSON output directory for a kernel type.
        Creates the directory if it doesn't exist.

        Args:
            kernel_type: Kernel type (e.g., "gemm", "attention")

        Returns:
            Path to JSON output directory (guaranteed to exist)
        """
        path = self.results / "json" / kernel_type
        return self._ensure_dir(path)

    # Compiled artifact paths

    def kernel_base_for(self, kernel_type: str, backend: str) -> Path:
        """
        Get base directory for kernel artifacts.
        Creates the directory if it doesn't exist.

        Args:
            kernel_type: Kernel type (e.g., "gemm", "attention")
            backend: Backend name (e.g., "wave", "iree")

        Returns:
            Path to kernel base directory (guaranteed to exist)
        """
        path = self.kernels / kernel_type / backend
        return self._ensure_dir(path)

    def mlir_for(self, kernel_type: str, backend: str) -> Path:
        """
        Get MLIR output directory.
        Creates the directory if it doesn't exist.

        Args:
            kernel_type: Kernel type
            backend: Backend name

        Returns:
            Path to MLIR directory (guaranteed to exist)
        """
        path = self.kernel_base_for(kernel_type, backend) / "mlir"
        return self._ensure_dir(path)

    def vmfb_for(self, kernel_type: str, backend: str) -> Path:
        """
        Get VMFB (compiled binary) output directory.
        Creates the directory if it doesn't exist.

        Args:
            kernel_type: Kernel type
            backend: Backend name

        Returns:
            Path to VMFB directory (guaranteed to exist)
        """
        path = self.kernel_base_for(kernel_type, backend) / "vmfb"
        return self._ensure_dir(path)

    # Tuning paths

    def tuning_base(self) -> Path:
        """
        Get base directory for all tuning results.
        Creates the directory if it doesn't exist.

        Returns:
            Path to tuning base directory (guaranteed to exist)
        """
        path = self.results / "tuning"
        return self._ensure_dir(path)

    def tuning_for(self, kernel_type: str) -> Path:
        """
        Get tuning results directory for a kernel type.
        Creates the directory if it doesn't exist.

        Args:
            kernel_type: Kernel type

        Returns:
            Path to tuning directory (guaranteed to exist)
        """
        path = self.tuning_base() / kernel_type
        return self._ensure_dir(path)

    # Debug dump paths

    def dump_for(self, *path_parts: str) -> Optional[Path]:
        """
        Get debug dump file path with flexible path structure.
        Creates parent directories automatically.

        Args:
            *path_parts: Variable path components (e.g., "iree", "log", "file.txt")

        Returns:
            Path to dump file, or None if dumps disabled

        Examples:
            dump_for("iree", "kernel.mlir")  # -> dumps/iree/kernel.mlir
            dump_for("iree", "log", "error.txt")  # -> dumps/iree/log/error.txt
        """
        if self.dumps is None:
            return None

        path = self.dumps
        for part in path_parts:
            path = path / part

        # Create parent directory if it doesn't exist
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def dump_dir_for(self, backend: str) -> Optional[Path]:
        """
        Get debug dump directory for a backend.
        Creates the directory if it doesn't exist.

        Args:
            backend: Backend name

        Returns:
            Path to backend dump directory (guaranteed to exist), or None if dumps disabled
        """
        if self.dumps is None:
            return None
        path = self.dumps / backend
        return self._ensure_dir(path)

    # Factory methods

    @classmethod
    def default(cls) -> "PathConfig":
        """
        Create default path configuration rooted at current workspace.

        Returns:
            PathConfig with default paths relative to current directory
        """
        workspace = Path.cwd()
        return cls(
            workspace=workspace,
            results=workspace / "results",
            kernels=workspace / "results" / "kernels",
            dumps=workspace / "dump",
        )

    @classmethod
    def from_workspace(
        cls, workspace: Path, enable_dumps: bool = False
    ) -> "PathConfig":
        """
        Create path configuration from a workspace directory.

        Args:
            workspace: Root workspace directory
            enable_dumps: Whether to enable debug dumps

        Returns:
            PathConfig rooted at the workspace
        """
        workspace = Path(workspace)
        return cls(
            workspace=workspace,
            results=workspace / "results",
            kernels=workspace / "results" / "kernels",
            dumps=workspace / "dumps" if enable_dumps else None,
        )

    @classmethod
    def for_testing(cls, tmp_dir: Path) -> "PathConfig":
        """
        Create isolated path configuration for testing.

        Args:
            tmp_dir: Temporary directory for test isolation

        Returns:
            PathConfig with all paths under tmp_dir
        """
        tmp_dir = Path(tmp_dir)
        return cls(
            workspace=tmp_dir,
            results=tmp_dir / "results",
            kernels=tmp_dir / "kernels",
            dumps=tmp_dir / "dumps",
        )

    @classmethod
    def from_legacy(
        cls, kernel_dir: Path, dump_dir: Optional[Path] = None
    ) -> "PathConfig":
        """
        Create PathConfig from legacy kernel_dir/dump_dir parameters.

        This is for backwards compatibility with old code that passes
        kernel_dir and dump_dir separately.

        Args:
            kernel_dir: Legacy kernel directory path
            dump_dir: Legacy dump directory path

        Returns:
            PathConfig that approximates the old behavior
        """
        kernel_dir = Path(kernel_dir)

        # Try to infer workspace from kernel_dir structure
        # kernel_dir is usually "results/kernels" or similar
        if kernel_dir.name == "kernels" and kernel_dir.parent.name == "results":
            workspace = kernel_dir.parent.parent
            results = kernel_dir.parent
        else:
            workspace = Path.cwd()
            results = workspace / "results"

        return cls(
            workspace=workspace,
            results=results,
            kernels=kernel_dir,
            dumps=Path(dump_dir) if dump_dir else None,
        )

    # Utility methods

    def ensure_dirs(self):
        """Create all base directories if they don't exist."""
        self.results.mkdir(parents=True, exist_ok=True)
        self.kernels.mkdir(parents=True, exist_ok=True)
        if self.dumps is not None:
            self.dumps.mkdir(parents=True, exist_ok=True)

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "workspace": str(self.workspace),
            "results": str(self.results),
            "kernels": str(self.kernels),
            "dumps": str(self.dumps) if self.dumps else None,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "PathConfig":
        """Create from dictionary."""
        return cls(
            workspace=Path(data["workspace"]),
            results=Path(data["results"]),
            kernels=Path(data["kernels"]),
            dumps=Path(data["dumps"]) if data.get("dumps") else None,
        )

    def __repr__(self) -> str:
        """String representation for debugging."""
        return (
            f"PathConfig(\n"
            f"  workspace={self.workspace},\n"
            f"  results={self.results},\n"
            f"  kernels={self.kernels},\n"
            f"  dumps={self.dumps}\n"
            f")"
        )
