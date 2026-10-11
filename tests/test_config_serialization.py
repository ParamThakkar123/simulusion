from simulusion.configs.dit_config import DiTConfig
from simulusion.configs.dreamer_config import DreamerConfig
from simulusion.configs.genie_config import GenieConfig, GenieSmallConfig
from simulusion.configs.iris_config import IRISConfig
from simulusion.configs.jepa_config import JEPAConfig


CONFIG_CASES = [
    (DreamerConfig, {"env": "cartpole_balance", "image_size": (32, 48)}),
    (JEPAConfig, {"model_name": "vit_tiny", "crop_scale": (0.5, 1.0)}),
    (IRISConfig, {"env": "ALE/Breakout-v5", "frame_height": 32}),
    (DiTConfig, {"BATCH": 8, "IMG_SIZE": 16}),
    (GenieConfig, {"num_frames": 4, "image_size": 16}),
    (GenieSmallConfig, {"num_frames": 4, "image_size": 16}),
]


def test_model_configs_yaml_roundtrip(tmp_path):
    for config_cls, overrides in CONFIG_CASES:
        config = config_cls.from_dict(overrides)
        path = tmp_path / f"{config_cls.__name__}.yaml"
        yaml_text = config.to_yaml(path)

        from_path = config_cls.from_yaml(path)
        from_text = config_cls.from_yaml(yaml_text)

        assert from_path.to_dict() == config.to_dict()
        assert from_text.to_dict() == config.to_dict()


def test_jepa_exposes_nested_training_dict_and_roundtrips():
    config = JEPAConfig.from_dict({"model_name": "vit_tiny"})
    nested = config.to_dict()

    assert nested["meta"]["model_name"] == "vit_tiny"
    assert config.to_train_dict()["meta"]["model_name"] == "vit_tiny"
    assert config.to_nested_dict()["meta"]["model_name"] == "vit_tiny"
    assert JEPAConfig.from_dict(nested).model_name == "vit_tiny"
