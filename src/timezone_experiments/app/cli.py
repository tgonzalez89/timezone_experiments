"""Unified command-line interface for experiments, data, and the dashboard."""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import subprocess
import sys
from pathlib import Path

from timezone_experiments import __version__
from timezone_experiments.app.errors import USER_ERRORS, describe_error
from timezone_experiments.core.config import TERRITORY_SETTINGS, ConfigError, ExperimentConfig, load_config
from timezone_experiments.data.boundaries import load_territory_geometries
from timezone_experiments.data.download.catalog import data_credits, input_record
from timezone_experiments.data.paths import DEFAULT_DATA_DIRECTORY, cache_directory, downloads_directory
from timezone_experiments.data.preparation import (
    InputOptions,
    load_experiment_territories,
    prepare_boundaries,
    prepare_inputs,
)
from timezone_experiments.experiments.results_io import (
    input_records,
    provenance_credits,
    read_results_config,
    read_results_csv,
    read_results_provenance,
    results_config_path,
    write_results_provenance,
)
from timezone_experiments.experiments.runner import run_experiment
from timezone_experiments.maps import (
    MapOptions,
    export_maps,
    geometry_feature_collection,
    result_feature_collection,
)

logger = logging.getLogger(__name__)


DEFAULT_RESULTS = Path("outputs/csv/results.csv")
DEFAULT_MAPS = Path("outputs/maps")
DEFAULT_OUTPUTS = Path("outputs")


def build_parser() -> argparse.ArgumentParser:
    """Build the single public command-line interface.

    Every command lists its options in the same order: ``--config``, ``--data-directory``, the command's own
    options, then (for run and dashboard) the "Optional input overrides" group. Defaults are stated in each help
    text. Which countries and administrative levels are used comes only from the configuration.
    """
    parser = argparse.ArgumentParser(
        prog="timezone_experiments",
        description="Explore timezone assignments; required input data is downloaded and cached automatically.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    parser.add_argument("--debug", action="store_true", help="Show full tracebacks instead of short error messages.")
    commands = parser.add_subparsers(dest="command", metavar="<command>", title="commands")

    run = _add_command(
        commands,
        "run",
        "Prepare data and run an experiment.",
        "Download or reuse the inputs the configuration needs, optimize every territory, and write a result CSV "
        "plus the configuration used (RESULTS with a .config.json suffix); add --maps to also export maps. Use the "
        "clean command to force fresh downloads or caches.",
    )
    _add_common_options(run)
    _add_results_option(run, "Where to write the result CSV")
    run.add_argument("--maps", action="store_true", help="Also export PNG, SVG, and HTML maps to --maps-dir.")
    _add_maps_dir_option(run, default=None)
    _add_input_overrides(run)

    maps = _add_command(
        commands,
        "maps",
        "Render maps from a saved result CSV.",
        "Redraw maps from a result CSV without re-running sampling, overlays, or optimization. Without --config, "
        "the configuration saved beside the CSV by run is used. A --config may change map settings, but its "
        "territory settings (countries and administrative levels) must match the saved ones.",
    )
    _add_common_options(maps, config_default="the configuration saved beside --results, else built-in settings")
    _add_results_option(maps, "Result CSV written by the run command")
    _add_maps_dir_option(maps, default=DEFAULT_MAPS)

    download = _add_command(
        commands,
        "download",
        "Download inputs ahead of time.",
        "Download every input the configuration needs, so a later run works offline: boundaries for the "
        "configured countries and levels, timezone boundaries, and the population grid for population coverage.",
    )
    _add_common_options(download)
    download.add_argument("--refresh", action="store_true", help="Download again even when a cached copy exists.")

    clean = _add_command(
        commands,
        "clean",
        "Remove downloads, caches, or generated outputs.",
        "Delete the selected targets. Removing downloads or caches makes the next run fetch or rebuild them.",
    )
    clean.add_argument(
        "--data-directory",
        type=Path,
        default=DEFAULT_DATA_DIRECTORY,
        metavar="DIR",
        help=f"Data root containing downloads/ and cache/ (default: {DEFAULT_DATA_DIRECTORY.as_posix()}).",
    )
    clean.add_argument(
        "--output-directory",
        type=Path,
        default=DEFAULT_OUTPUTS,
        metavar="DIR",
        help=f"Output root containing csv/ and maps/ (default: {DEFAULT_OUTPUTS.as_posix()}).",
    )
    clean.add_argument("--dry-run", action="store_true", help="List what would be removed without deleting it.")
    targets = clean.add_argument_group("Targets (choose at least one)")
    targets.add_argument(
        "--downloads", action="store_true", help="Remove downloaded input files (DATA_DIRECTORY/downloads)."
    )
    targets.add_argument(
        "--cache", action="store_true", help="Remove prepared-input and map caches (DATA_DIRECTORY/cache)."
    )
    targets.add_argument("--csv", action="store_true", help="Remove result CSVs (OUTPUT_DIRECTORY/csv).")
    targets.add_argument("--maps", action="store_true", help="Remove exported maps (OUTPUT_DIRECTORY/maps).")
    targets.add_argument("--all", action="store_true", help="Remove downloads, caches, and the whole output root.")

    dashboard = _add_command(
        commands,
        "dashboard",
        "Open the interactive dashboard.",
        "Launch the Streamlit dashboard in a browser. The configuration provides the initial values of every "
        "control; inputs are downloaded and cached automatically.",
    )
    _add_common_options(dashboard)
    dashboard.add_argument(
        "--port", type=_positive_integer, default=8501, help="Local port to serve the dashboard on (default: 8501)."
    )
    dashboard.add_argument("--headless", action="store_true", help="Do not open a browser automatically.")
    _add_input_overrides(dashboard)
    return parser


def _add_command(
    commands: argparse._SubParsersAction[argparse.ArgumentParser], name: str, summary: str, description: str
) -> argparse.ArgumentParser:
    return commands.add_parser(name, help=summary, description=f"{summary} {description}")


def _add_common_options(
    parser: argparse.ArgumentParser,
    config_default: str = "built-in settings, as in examples/default.json",
) -> None:
    parser.add_argument(
        "--config",
        type=Path,
        metavar="JSON",
        help=f"Experiment configuration (default: {config_default}).",
    )
    parser.add_argument(
        "--data-directory",
        type=Path,
        default=DEFAULT_DATA_DIRECTORY,
        metavar="DIR",
        help=(
            "Data root; downloads are stored and reused in DIR/downloads, caches in DIR/cache "
            f"(default: {DEFAULT_DATA_DIRECTORY.as_posix()})."
        ),
    )


def _add_results_option(parser: argparse.ArgumentParser, purpose: str) -> None:
    parser.add_argument(
        "--results",
        type=Path,
        default=DEFAULT_RESULTS,
        metavar="CSV",
        help=f"{purpose} (default: {DEFAULT_RESULTS.as_posix()}).",
    )


def _add_maps_dir_option(parser: argparse.ArgumentParser, *, default: Path | None) -> None:
    # run leaves the default unset so that --maps-dir without --maps can be reported instead of silently ignored.
    parser.add_argument(
        "--maps-dir",
        type=Path,
        default=default,
        metavar="DIR",
        help=f"Directory for the PNG, SVG, and HTML map files (default: {DEFAULT_MAPS.as_posix()}).",
    )


def _add_input_overrides(parser: argparse.ArgumentParser) -> None:
    overrides = parser.add_argument_group("Optional input overrides (normally automatic)")
    overrides.add_argument(
        "--population",
        type=Path,
        metavar="CSV",
        help="Use this population CSV (latitude, longitude, population) instead of the GHSL download.",
    )
    overrides.add_argument(
        "--timezones",
        type=Path,
        metavar="GEOJSON",
        help="Use this timezone-boundary file instead of the timezone-boundary-builder download.",
    )


def input_options(args: argparse.Namespace) -> InputOptions:
    """Translate the data options of run/dashboard into data preparation settings."""
    return InputOptions(
        data_directory=args.data_directory,
        population_path=args.population,
        timezones_path=args.timezones,
    )


def _positive_integer(value: str) -> int:
    result = int(value)
    if result < 1:
        msg = "must be a positive integer"
        raise argparse.ArgumentTypeError(msg)
    return result


def cmd_run(args: argparse.Namespace) -> int:
    """Prepare required data, optimize territories, and export results."""
    if args.maps_dir is not None and not args.maps:
        msg = "--maps-dir only applies when maps are exported; add --maps."
        raise ValueError(msg)
    logger.info("Loading configuration: %s", args.config or "built-in defaults")
    config = load_config(args.config)
    options = input_options(args)
    inputs = prepare_inputs(config, options)
    territories = load_experiment_territories(config, options, inputs)
    results = run_experiment(config, territories, output_path=args.results)
    records = input_records(inputs)
    write_results_provenance(args.results, records)
    print(f"Wrote {len(results)} territory assignments to {args.results}.")
    if args.maps:
        maps_dir = args.maps_dir or DEFAULT_MAPS
        attribution = data_credits(records.values())
        export_maps(
            result_feature_collection(territories, results),
            maps_dir,
            MapOptions.from_config(
                config, cache_directory=cache_directory(args.data_directory), attribution=attribution
            ),
        )
        print(f"Wrote map exports to {maps_dir}.")
    return 0


def cmd_download(args: argparse.Namespace) -> int:
    """Cache every input the configuration needs."""
    config = load_config(args.config)
    inputs = prepare_inputs(config, InputOptions(data_directory=args.data_directory, refresh=args.refresh))
    for label, path in (
        ("Boundaries", inputs.boundaries),
        ("Population", inputs.population),
        ("Timezones", inputs.timezones),
    ):
        if path is not None:
            print(f"{label}: {path}")
    print("Data is ready. Run an experiment with the same configuration to use it.")
    return 0


def cmd_maps(args: argparse.Namespace) -> int:
    """Render saved assignments using only administrative boundary geometry."""
    results = read_results_csv(args.results)
    config = _maps_config(args)
    boundaries = prepare_boundaries(config, InputOptions(data_directory=args.data_directory))
    geometries = load_territory_geometries(boundaries, config)
    export_maps(
        geometry_feature_collection(geometries, results),
        args.maps_dir,
        MapOptions.from_config(
            config,
            cache_directory=cache_directory(args.data_directory),
            attribution=_maps_attribution(args, config, boundaries),
        ),
    )
    print(f"Wrote map exports for {len(results)} saved territory assignments to {args.maps_dir}.")
    return 0


def _maps_attribution(args: argparse.Namespace, config: ExperimentConfig, boundaries: Path) -> str:
    """Credit the data recorded with the results; without a record, the boundaries drawn and the usual inputs."""
    provenance = read_results_provenance(args.results)
    if provenance is not None:
        return provenance_credits(provenance)
    extra = ["timezone_boundary_builder"] + (["ghsl_population"] if config.coverage_type == "population" else [])
    return data_credits([input_record(boundaries)], extra_sources=extra)


def _maps_config(args: argparse.Namespace) -> ExperimentConfig:
    """Use the configuration saved with the results, or check that --config defines the same territories."""
    saved = read_results_config(args.results)
    if args.config is None:
        if saved is None:
            logger.info("No saved configuration beside %s; using built-in defaults.", args.results)
            return load_config(None)
        logger.info("Using the configuration saved with the results: %s", results_config_path(args.results))
        return saved
    config = load_config(args.config)
    if saved is None:
        return config
    given, expected = config.to_mapping(), saved.to_mapping()
    differences = [
        f"{name} ({json.dumps(given[name])} instead of {json.dumps(expected[name])})"
        for name in TERRITORY_SETTINGS
        if given[name] != expected[name]
    ]
    if differences:
        msg = (
            f"{args.config} defines different territories than the run that wrote {args.results}: "
            f"{'; '.join(differences)}. Omit --config to use the saved configuration "
            f"({results_config_path(args.results)}), or change only map settings."
        )
        raise ConfigError(msg)
    return config


def cmd_clean(args: argparse.Namespace) -> int:
    """Remove explicitly selected generated data and outputs."""
    paths: list[tuple[str, Path]] = []
    if args.all or args.downloads:
        paths.append(("downloads", downloads_directory(args.data_directory)))
    if args.all or args.cache:
        paths.append(("cache", cache_directory(args.data_directory)))
    if args.all:
        paths.append(("outputs", args.output_directory))
    else:
        paths.extend((name, args.output_directory / name) for name in ("csv", "maps") if getattr(args, name))
    if not paths:
        msg = "Select at least one clean target: --downloads, --cache, --csv, --maps, or --all."
        raise ValueError(msg)
    for label, path in paths:
        if not path.exists() and not path.is_symlink():
            print(f"Not found ({label}): {path}")
            continue
        if args.dry_run:
            print(f"Would remove ({label}): {path}")
            continue
        _remove_clean_path(path)
        print(f"Removed ({label}): {path}")
    return 0


def _remove_clean_path(path: Path) -> None:
    resolved = path.resolve()
    cwd = Path.cwd().resolve()
    if resolved == cwd or resolved in cwd.parents:
        msg = f"Refusing to remove a workspace or ancestor directory: {path}"
        raise ValueError(msg)
    if path.is_symlink() or path.is_file():
        path.unlink()
    else:
        shutil.rmtree(path)


def cmd_dashboard(args: argparse.Namespace) -> int:
    """Launch Streamlit through this CLI using the active Python environment."""
    load_config(args.config)
    command = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(Path(__file__).parent / "dashboard" / "app.py"),
        "--server.address",
        "localhost",
        "--server.port",
        str(args.port),
        "--server.headless",
        str(args.headless).lower(),
        "--",
        "--data-directory",
        str(args.data_directory),
    ]
    for option in ("config", "population", "timezones"):
        value = getattr(args, option)
        if value is not None:
            command.extend((f"--{option.replace('_', '-')}", str(value)))
    return subprocess.run(command, check=False).returncode  # noqa: S603


def main(argv: list[str] | None = None) -> int:
    """Dispatch one command, showing actionable errors without a traceback."""
    parser = build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    if args.command is None:
        parser.print_help()
        return 0
    handlers = {
        "run": cmd_run,
        "maps": cmd_maps,
        "download": cmd_download,
        "clean": cmd_clean,
        "dashboard": cmd_dashboard,
    }
    try:
        return handlers[args.command](args)
    except USER_ERRORS as error:
        if args.debug:
            raise
        parser.exit(2, f"{parser.prog}: error: {describe_error(error)}\n")
    except KeyboardInterrupt:
        print("Cancelled.", file=sys.stderr)
        return 130


if __name__ == "__main__":
    sys.exit(main())
