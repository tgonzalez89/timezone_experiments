"""Static and interactive maps of experiment results.

Each map statistic is a ``MapSpec`` (offset.py, shift.py, median_noon.py, solar_events.py) registered in
registry.py; static.py and interactive.py render any spec, and export.py runs the batch of jobs.
"""

from timezone_experiments.maps.export import export_maps
from timezone_experiments.maps.features import geometry_feature_collection, result_feature_collection
from timezone_experiments.maps.interactive import build_plotly_figure
from timezone_experiments.maps.registry import map_title
from timezone_experiments.maps.spec import MAP_FIELDS, MAP_KINDS, MapKind, MapOptions

__all__ = [
    "MAP_FIELDS",
    "MAP_KINDS",
    "MapKind",
    "MapOptions",
    "build_plotly_figure",
    "export_maps",
    "geometry_feature_collection",
    "map_title",
    "result_feature_collection",
]
