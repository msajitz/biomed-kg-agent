from biomed_kg_agent.log import setup_logger


def test_setup_logger() -> None:
    logger = setup_logger("test_logger")
    assert logger.name == "test_logger"
    assert logger.hasHandlers()
