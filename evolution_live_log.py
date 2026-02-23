"""
Live running log for evolution (CLI and GUI).

Writes timestamped lines to a file so you can see where you are (run, generation,
best fitness). File is flushed after each line so `tail -f evolution_live.log` works.
Optional: echo to console (CLI) or collect lines for display (GUI).
"""

import os
from datetime import datetime
from typing import Optional, List


class LiveLogger:
    """Write live evolution progress to a file (and optionally console or a list)."""

    def __init__(
        self,
        log_path: str,
        *,
        echo_console: bool = False,
        echo_lines: Optional[List[str]] = None,
    ):
        """
        Args:
            log_path: Path to the log file (e.g. results/evolution_XXX/evolution_live.log).
            echo_console: If True, also print each line to stdout (for CLI).
            echo_lines: If provided, append each line to this list (for GUI display).
        """
        self.log_path = log_path
        self.echo_console = echo_console
        self.echo_lines = echo_lines if echo_lines is not None else []
        self._file = None
        self._open()

    def _open(self) -> None:
        if not self.log_path:
            return
        os.makedirs(os.path.dirname(self.log_path) or ".", exist_ok=True)
        self._file = open(self.log_path, "w", encoding="utf-8")

    def _write(self, msg: str) -> None:
        line = f"[{datetime.now().strftime('%H:%M:%S')}] {msg}"
        if self._file is not None:
            self._file.write(line + "\n")
            self._file.flush()
        if self.echo_console:
            print(line)
        if self.echo_lines is not None:
            self.echo_lines.append(line)

    def log(self, msg: str) -> None:
        """Log a single message."""
        self._write(msg)

    def log_run_start(self, run_index: int, n_runs: int, seed: int) -> None:
        """Log start of a run."""
        self._write(f"Run {run_index + 1}/{n_runs} (seed={seed}) — start")

    def log_run_end(self, run_index: int, n_runs: int) -> None:
        """Log end of a run."""
        self._write(f"Run {run_index + 1}/{n_runs} — done")

    def log_gen(
        self,
        run_index: int,
        n_runs: int,
        gen: int,
        ngen: int,
        record: dict,
        best_phenotype: str = "",
    ) -> None:
        """Log one generation (call from on_generation_callback). gen is 0-based; ngen is total."""
        best_mae = record.get("min")
        test_mae = record.get("fitness_test")
        invalid = record.get("invalid", 0)
        try:
            best_str = f" | best_mae={float(best_mae):.4f}" if best_mae is not None else ""
        except (TypeError, ValueError):
            best_str = ""
        try:
            test_str = f" | test_mae={float(test_mae):.4f}" if test_mae is not None else ""
        except (TypeError, ValueError):
            test_str = ""
        inv_str = f" | invalid={invalid}" if invalid else ""
        pheno_snippet = (best_phenotype[:60] + "…") if len(best_phenotype) > 60 else best_phenotype
        pheno_str = f" | {pheno_snippet}" if pheno_snippet else ""
        self._write(
            f"Run {run_index + 1}/{n_runs} | Gen {gen + 1}/{ngen}{best_str}{test_str}{inv_str}{pheno_str}"
        )

    def close(self) -> None:
        """Close the log file."""
        if self._file is not None:
            try:
                self._file.close()
            except Exception:
                pass
            self._file = None

    def __enter__(self) -> "LiveLogger":
        return self

    def __exit__(self, *args) -> None:
        self.close()
