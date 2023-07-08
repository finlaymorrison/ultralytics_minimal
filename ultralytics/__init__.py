# Ultralytics YOLO 🚀, AGPL-3.0 license

__version__ = '8.0.126'

from ultralytics.yolo.engine.model import YOLO
from ultralytics.yolo.utils.checks import check_yolo as checks
from ultralytics.yolo.utils.downloads import download

__all__ = '__version__', 'YOLO', 'checks', 'download'  # allow simpler import
