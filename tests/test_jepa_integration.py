import pytest
from simulusion.configs.jepa_config import JEPAConfig


@pytest.mark.integration
def test_jepa_construct_no_crash(tmp_path):
    from simulusion.models.jepa_agent import JEPAAgent

    config = JEPAConfig()
    config.folder = str(tmp_path / "jepa_integration_test")
    config.epochs = 1
    config.warmup = 0
    config.batch_size = 1
    config.num_workers = 0
    config.dataset = "imagefolder"
    config.root_path = str(tmp_path)

    agent = JEPAAgent(config)
    assert agent is not None
