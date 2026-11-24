"""The Streamlit script of the dashboard: page layout and running an experiment from the current controls."""

from __future__ import annotations

import sys
import tempfile
import threading
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING

import streamlit as st

from timezone_experiments.app.cli import build_parser, input_options
from timezone_experiments.app.dashboard.inputs import dashboard_inputs
from timezone_experiments.app.dashboard.results import ExperimentRun, show_results
from timezone_experiments.app.dashboard.settings import build_config
from timezone_experiments.app.errors import USER_ERRORS, describe_error
from timezone_experiments.core.config import load_config
from timezone_experiments.data.paths import cache_directory
from timezone_experiments.data.preparation import load_experiment_territories
from timezone_experiments.experiments.runner import run_experiment

if TYPE_CHECKING:
    from types import ModuleType

    from timezone_experiments.app.dashboard.settings import DashboardInputs, UploadedFile
    from timezone_experiments.core.config import ExperimentConfig
    from timezone_experiments.data.preparation import InputOptions

# Sessions share the download and cache directories, so only one experiment prepares inputs at a time.
_RUN_LOCK = threading.Lock()
_LAST_RUN = "last_experiment_run"


def main(argv: list[str] | None = None) -> None:
    """Render the interactive experiment dashboard."""
    st.set_page_config(page_title="Timezone Experiments", layout="wide")
    st.title("Hypothetical timezone experiments")
    st.caption("Explore permanent UTC offsets using configurable solar-noon and coverage rules.")
    args = build_parser().parse_args(["dashboard", *(argv or [])])
    try:
        config = load_config(args.config)
        st.info("Required data is downloaded automatically and cached. Population and timezone uploads are optional.")
        inputs = dashboard_inputs(st, config)
        if st.button("Run experiment", type="primary"):
            _execute_experiment(st, inputs, config, input_options(args))
        run = st.session_state.get(_LAST_RUN)
        if run is not None:
            show_results(st, run, inputs.map_field, cache_directory(args.data_directory))
    except USER_ERRORS as error:
        st.error(describe_error(error))


def _execute_experiment(
    st: ModuleType,
    inputs: DashboardInputs,
    base_config: ExperimentConfig,
    options: InputOptions,
) -> None:
    if not _RUN_LOCK.acquire(blocking=False):
        st.warning("Another experiment is running on this server; try again when it finishes.")
        return
    try:
        config = build_config(inputs, base_config)
        with tempfile.TemporaryDirectory() as temporary_directory:
            directory = Path(temporary_directory)
            options = replace(
                options,
                population_path=_uploaded_path(
                    inputs.population_upload, directory / "population.csv", options.population_path
                ),
                timezones_path=_uploaded_path(
                    inputs.timezone_upload, directory / "timezones.geojson", options.timezones_path
                ),
            )
            st.info("Preparing input data. Initial downloads may take several minutes.")
            territories = load_experiment_territories(config, options)
            progress = st.progress(0, text="Optimizing territories...")

            def update_progress(index: int, total: int, territory_id: str) -> None:
                progress.progress(index / total, text=f"Optimizing {territory_id} ({index}/{total})")

            results = run_experiment(config, territories, on_territory_done=update_progress)
        st.session_state[_LAST_RUN] = ExperimentRun(config, territories, results)
        st.success(f"Optimized {len(results)} territories.")
    except USER_ERRORS as error:
        st.error(describe_error(error))
    finally:
        _RUN_LOCK.release()


def _uploaded_path(upload: UploadedFile | None, path: Path, fallback: Path | None) -> Path | None:
    if upload is None:
        return fallback
    path.write_bytes(upload.getvalue())
    return path


if __name__ == "__main__":
    main(sys.argv[1:])
