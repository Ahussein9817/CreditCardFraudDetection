"""Slow smoke test: renders every page of app/app.py headlessly (Streamlit AppTest).

Needs the Git LFS model files (`git lfs pull`), so it is skipped unless RUN_SLOW=1.
    RUN_SLOW=1 pytest tests/test_app_smoke.py
"""
import os
from pathlib import Path

import pytest

APP = Path(__file__).resolve().parent.parent / "app" / "app.py"
pytestmark = pytest.mark.skipif(os.environ.get("RUN_SLOW") != "1", reason="set RUN_SLOW=1 to run")


def _pages():
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file(str(APP), default_timeout=300).run()
    return at, list(at.sidebar.radio[0].options)


def test_every_page_renders_without_exception():
    at, pages = _pages()
    assert len(pages) == 6
    assert not at.exception, [e.value for e in at.exception]
    for page in pages:
        at.sidebar.radio[0].set_value(page).run()
        assert not at.exception, (page, [e.value for e in at.exception])
