from __future__ import annotations

import subprocess
import sys


def _assert_import_does_not_load(module: str, forbidden: tuple[str, ...]) -> None:
    code = (
        "import sys; "
        f"import {module}; "
        f"forbidden={forbidden!r}; "
        "loaded=sorted(name for name in sys.modules "
        "if any(name == item or name.startswith(item + '.') "
        "for item in forbidden)); "
        "assert not loaded, loaded"
    )
    subprocess.run([sys.executable, "-c", code], check=True)


def test_utils_import_does_not_load_napari_or_ml_stack() -> None:
    _assert_import_does_not_load(
        "cns_control.utils",
        ("napari", "raman_mda_engine", "cellpose", "torch"),
    )
