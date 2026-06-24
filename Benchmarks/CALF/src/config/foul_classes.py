"""Re-export foul-only vocabulary from repo-root config/foul_classes.py."""

import importlib.util
from pathlib import Path

_SN_ROOT = Path(__file__).resolve().parents[4]
_FOUL_CLASSES_PATH = _SN_ROOT / "config" / "foul_classes.py"
_spec = importlib.util.spec_from_file_location("sn_foul_classes", _FOUL_CLASSES_PATH)
if _spec is None or _spec.loader is None:
    raise ImportError(f"Unable to load foul classes from {_FOUL_CLASSES_PATH}")
_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)
globals().update(
    {name: value for name, value in _module.__dict__.items() if not name.startswith("_")}
)
