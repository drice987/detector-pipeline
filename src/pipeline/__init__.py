from importlib.metadata import PackageNotFoundError, version
from .core import AutomatedDetectorPipeline, DataFileHandler
from .io import read_detector_file, valid_extensions
from .rendering import DashboardRenderer
from .stats import EMAStats, WelfordRollingStats

try:
    __version__ = version("pipeline")
except PackageNotFoundError:
    __version__ = "unknown"

__all__ = [
    "AutomatedDetectorPipeline",
    "DataFileHandler",
    "read_detector_file",
    "valid_extensions",
    "WelfordRollingStats",
    "EMAStats",
    "DashboardRenderer",
]