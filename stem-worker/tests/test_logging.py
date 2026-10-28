import logging

from stem_worker.main import configure_logging


def test_rich_logs_are_written_to_stderr_not_stdout(capsys) -> None:
    configure_logging()

    logging.getLogger("stem_worker.test").info("worker is ready")

    captured = capsys.readouterr()
    assert captured.out == ""
    assert "INFO" in captured.err
    assert "worker is ready" in captured.err
