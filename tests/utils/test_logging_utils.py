"""Tests for ``simulusion.utils.logging_utils.setup_logging``."""

import logging

import pytest

from simulusion.utils.logging_utils import setup_logging


@pytest.fixture
def restore_loggers():
    """Undo setup_logging's changes to the package and root loggers."""
    package = logging.getLogger("simulusion")
    root = logging.getLogger()
    saved = (package.handlers[:], package.propagate, package.level, root.handlers[:])
    yield root
    package.handlers[:], package.propagate, package.level, root.handlers[:] = saved


def test_records_still_reach_root_handlers(restore_loggers, caplog):
    """Constructing an agent calls setup_logging("simulusion"); that must not cut
    simulusion.* records off from the application's (or caplog's) root handlers."""
    setup_logging("simulusion")

    with caplog.at_level("WARNING", logger="simulusion.utils.device"):
        logging.getLogger("simulusion.utils.device").warning("still visible")

    assert "still visible" in caplog.text


def test_console_handler_only_when_root_has_none(restore_loggers):
    root = restore_loggers

    root.handlers[:] = []
    assert len(setup_logging("simulusion").handlers) == 1

    root.handlers[:] = [logging.NullHandler()]
    assert setup_logging("simulusion").handlers == []


def test_file_handler_is_attached(restore_loggers, tmp_path):
    log_file = tmp_path / "logs" / "run.log"
    logger = setup_logging("simulusion", log_file=str(log_file))
    logger.warning("to file")
    for handler in logger.handlers:
        handler.flush()
        handler.close()
    assert "to file" in log_file.read_text()
