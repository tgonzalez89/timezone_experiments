"""Unified command-line interface for experiments, data, and the dashboard."""

from __future__ import annotations

import argparse
import logging
import shutil
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from typing import override
from zipfile import BadZipFile

from timezone_experiments import __version__
from timezone_experiments.config import load_config
from timezone_experiments.data.boundaries import load_territory_geometries
from timezone_experiments.data.download_boundaries import download
from timezone_experiments.data.paths import DEFAULT_DOWNLOAD_DIRECTORY, cache_directory_for
from timezone_experiments.data.preparation import (
    InputOptions,
    load_experiment_territories,
    prepare_boundaries,
    prepare_inputs,
)
from timezone_experiments.experiments.runner import read_results_csv, run_experiment
from timezone_experiments.visualization.maps import export_maps, export_result_maps

logger = logging.getLogger(__name__)


class DefaultsHelpFormatter(argparse.ArgumentDefaultsHelpFormatter):
    """Display meaningful automatic defaults instead of opaque None values."""

    @override
    def _get_help_string(self, action: argparse.Action) -> str | None:
        text = super()._get_help_string(action)
        automatic = {
            "config": "built-in experiment settings",
            "boundaries": "automatic, according to configuration",
            "population": "automatic when configuration needs population",
            "timezones": "automatic",
            "source_admin_level": "detected from downloaded data",
            "population_cell_degrees": "configuration value, initially 1.0",
        }
        if text is not None and action.default is None:
            text = text.replace("%(default)s", automatic.get(action.dest, "none"))
        return text


def build_parser() -> argparse.ArgumentParser:
    """Build the single public command-line interface."""
    parser = argparse.ArgumentParser(
        prog="timezone_experiments",
        description="Explore timezone assignments; required input data is downloaded and cached automatically.",
        formatter_class=DefaultsHelpFormatter,
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    commands = parser.add_subparsers(dest="command", metavar="<command>")
    run = commands.add_parser("run", help="Prepare data and run an experiment.", formatter_class=DefaultsHelpFormatter)
    _add_input_options(run)
    run.add_argument("--output", type=Path, default=Path("outputs/csv/results.csv"), help="Result CSV path.")
    run.add_argument(
        "--maps",
        action="store_true",
        help="Export static and interactive maps using configured workers.",
    )

    maps = commands.add_parser(
        "maps",
        help="Generate maps from a saved result CSV without rerunning calculations.",
        formatter_class=DefaultsHelpFormatter,
    )
    maps.add_argument("--results", type=Path, default=Path("outputs/csv/results.csv"), help="Saved result CSV.")
    maps.add_argument("--output", type=Path, default=Path("outputs/maps"), help="Map output directory.")
    maps.add_argument("--config", type=Path, help="Original experiment configuration; also sets map workers.")
    maps.add_argument("--data-directory", type=Path, default=DEFAULT_DOWNLOAD_DIRECTORY, help="Download directory.")
    maps.add_argument("--boundaries", type=Path, help="Original administrative boundary GeoJSON.")
    maps.add_argument("--source-admin-level", type=int, help="Administrative level for an external boundary file.")

    data = commands.add_parser(
        "download",
        help="Pre-download configured inputs without running an experiment.",
        formatter_class=DefaultsHelpFormatter,
    )
    _add_input_options(data, paths=False)
    data.add_argument("--population", action="store_true", help="Additionally cache population inputs.")
    data.add_argument("--include-antarctica", action="store_true", help="Additionally include Antarctica.")
    data.add_argument(
        "--population-cell-degrees",
        type=float,
        choices=(0.1, 0.25, 0.5, 1.0),
        help="Override the configured population CSV resolution for this download.",
    )

    boundaries = commands.add_parser(
        "download-boundaries",
        help="Download one specific administrative dataset.",
        formatter_class=DefaultsHelpFormatter,
    )
    boundaries.add_argument(
        "--source", choices=("gadm", "geoboundaries", "naturalearth"), default="gadm", help="Provider."
    )
    boundaries.add_argument("--country", default="USA", help="Three-letter country code.")
    boundaries.add_argument("--level", type=int, default=1, help="Administrative level; Natural Earth requires 0.")
    boundaries.add_argument(
        "--data-directory",
        type=Path,
        default=DEFAULT_DOWNLOAD_DIRECTORY,
        help="Download directory.",
    )
    boundaries.add_argument("--refresh", action="store_true", help="Download again even when cached.")

    clean = commands.add_parser(
        "clean",
        help="Remove selected downloads, caches, or generated outputs.",
        formatter_class=DefaultsHelpFormatter,
    )
    clean.add_argument("--downloads", action="store_true", help="Remove downloaded input files.")
    clean.add_argument("--cache", action="store_true", help="Remove prepared and map caches.")
    clean.add_argument("--csv", action="store_true", help="Remove generated CSV outputs.")
    clean.add_argument("--maps", action="store_true", help="Remove generated map outputs.")
    clean.add_argument("--outputs", action="store_true", help="Remove all generated outputs (CSV and maps).")
    clean.add_argument("--all", action="store_true", help="Remove downloads, caches, and all outputs.")
    clean.add_argument("--dry-run", action="store_true", help="List selected paths without deleting them.")
    clean.add_argument(
        "--data-directory",
        type=Path,
        default=DEFAULT_DOWNLOAD_DIRECTORY,
        help="Downloaded-input directory to clean.",
    )
    clean.add_argument("--cache-directory", type=Path, help="Cache directory; defaults beside downloads.")
    clean.add_argument(
        "--output-directory",
        type=Path,
        default=Path("outputs"),
        help="Root containing csv/ and maps/.",
    )

    dashboard = commands.add_parser(
        "dashboard",
        help="Open the interactive dashboard with automatic data preparation.",
        formatter_class=DefaultsHelpFormatter,
    )
    _add_input_options(dashboard)
    dashboard.add_argument("--port", type=_positive_integer, default=8501, help="Local dashboard port.")
    dashboard.add_argument("--headless", action="store_true", help="Do not open a browser automatically.")
    return parser


def _add_input_options(parser: argparse.ArgumentParser, *, paths: bool = True) -> None:
    parser.add_argument("--config", type=Path, help="Experiment configuration JSON.")
    parser.add_argument("--data-directory", type=Path, default=DEFAULT_DOWNLOAD_DIRECTORY, help="Download directory.")
    parser.add_argument(
        "--refresh", action="store_true", help="Refresh required downloads instead of reusing the cache."
    )
    if paths:
        overrides = parser.add_argument_group("Optional input overrides (normally automatic)")
        overrides.add_argument("--boundaries", type=Path, help="Use an existing administrative boundary GeoJSON.")
        overrides.add_argument("--population", type=Path, help="Use an existing population CSV.")
        overrides.add_argument("--timezones", type=Path, help="Use an existing timezone-boundary GeoJSON.")
        overrides.add_argument(
            "--source-admin-level", type=int, help="Administrative level for an external boundary file."
        )


def input_options(args: argparse.Namespace) -> InputOptions:
    """Translate shared command options into reusable data preparation settings."""
    return InputOptions(
        data_directory=args.data_directory,
        boundaries_path=args.boundaries,
        population_path=args.population,
        timezones_path=args.timezones,
        source_admin_level=args.source_admin_level,
        refresh=args.refresh,
    )


def _positive_integer(value: str) -> int:
    result = int(value)
    if result < 1:
        msg = "must be a positive integer"
        raise argparse.ArgumentTypeError(msg)
    return result


def cmd_run(args: argparse.Namespace) -> int:
    """Prepare required data, optimize territories, and export results."""
    logger.info("Loading configuration: %s", args.config or "built-in defaults")
    config = load_config(args.config)
    territories = load_experiment_territories(config, input_options(args))
    results = run_experiment(config, territories, output_path=args.output)
    print(f"Wrote {len(results)} territory assignments to {args.output}.")
    if args.maps:
        directory = args.output.parent.parent / "maps"
        export_maps(
            territories,
            results,
            directory,
            workers=config.workers,
            cache_directory=cache_directory_for(args.data_directory),
            event_map_cell_degrees=config.event_map_cell_degrees,
            map_geometry_simplification_degrees=config.map_geometry_simplification_degrees,
            rasterize_static_map_fills=config.rasterize_static_map_fills,
        )
        print(f"Wrote map exports to {directory}.")
    return 0


def cmd_download(args: argparse.Namespace) -> int:
    """Cache configured inputs, with optional requests to prepare extra datasets."""
    config = load_config(args.config)
    config = replace(
        config,
        coverage_type="population" if args.population else config.coverage_type,
        include_antarctica=args.include_antarctica or config.include_antarctica,
        population_cell_degrees=(
            config.population_cell_degrees if args.population_cell_degrees is None else args.population_cell_degrees
        ),
    )
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
    config = load_config(args.config)
    results = read_results_csv(args.results)
    options = InputOptions(
        data_directory=args.data_directory,
        boundaries_path=args.boundaries,
        source_admin_level=args.source_admin_level,
    )
    boundaries = prepare_boundaries(config, options)
    geometries = load_territory_geometries(boundaries, config, source_admin_level=args.source_admin_level)
    export_result_maps(
        geometries,
        results,
        args.output,
        workers=config.workers,
        cache_directory=cache_directory_for(args.data_directory),
        event_map_cell_degrees=config.event_map_cell_degrees,
        map_geometry_simplification_degrees=config.map_geometry_simplification_degrees,
        rasterize_static_map_fills=config.rasterize_static_map_fills,
    )
    print(f"Wrote map exports for {len(results)} saved territory assignments to {args.output}.")
    return 0


def cmd_download_boundaries(args: argparse.Namespace) -> int:
    """Cache an explicitly selected country/provider/level dataset."""
    path = args.data_directory / f"boundaries_{args.source}_{args.country.upper()}_{args.level}.geojson"
    if args.refresh or not path.is_file():
        path = download(args.source, args.country, args.level, args.data_directory)
    print(f"Boundaries: {path}")
    return 0


def cmd_clean(args: argparse.Namespace) -> int:
    """Remove explicitly selected generated data and outputs."""
    cache_directory = args.cache_directory or cache_directory_for(args.data_directory)
    clean_downloads = args.all or args.downloads
    clean_cache = args.all or args.cache
    clean_outputs = args.all or args.outputs
    clean_csv = clean_outputs or args.csv
    clean_maps = clean_outputs or args.maps
    paths: list[tuple[str, Path]] = []
    if clean_downloads:
        paths.append(("downloads", args.data_directory))
    if clean_cache:
        paths.append(("cache", cache_directory))
    if clean_csv and not clean_outputs:
        paths.append(("csv", args.output_directory / "csv"))
    if clean_maps and not clean_outputs:
        paths.append(("maps", args.output_directory / "maps"))
    if clean_outputs:
        paths.append(("outputs", args.output_directory))
    if not paths:
        msg = "Select at least one clean target: --downloads, --cache, --csv, --maps, --outputs, or --all."
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
        str(Path(__file__).with_name("dashboard.py")),
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
    for option in ("config", "boundaries", "population", "timezones", "source_admin_level"):
        value = getattr(args, option)
        if value is not None:
            command.extend((f"--{option.replace('_', '-')}", str(value)))
    if args.refresh:
        command.append("--refresh")
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
        "download-boundaries": cmd_download_boundaries,
        "clean": cmd_clean,
        "dashboard": cmd_dashboard,
    }
    try:
        return handlers[args.command](args)
    except (OSError, ValueError, TypeError, RuntimeError, BadZipFile) as error:
        parser.exit(2, f"{parser.prog}: error: {error}\n")
    except KeyboardInterrupt:
        print("Cancelled.", file=sys.stderr)
        return 130


if __name__ == "__main__":
    sys.exit(main())
