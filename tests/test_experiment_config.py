from dataclasses import dataclass

from simulusion.experiments import (
    dotlist_to_dict,
    instantiate_dataclass,
    load_experiment_config,
    update_config_object,
)


@dataclass
class DemoDataclassConfig:
    seed: int = 0
    preset: str | None = None
    width: int = 64

    def __post_init__(self):
        if self.preset == "small":
            self.width = 32


class DemoObjectConfig:
    def __init__(self):
        self.total_epochs = 600
        self.collect_epsilon = 0.1
        self.vocab_size = 512


def test_dotlist_to_dict_parses_nested_values():
    assert dotlist_to_dict(["optimization.lr=3e-4", "seed=7", "flag=true"]) == {
        "optimization": {"lr": 0.0003},
        "seed": 7,
        "flag": True,
    }


def test_load_experiment_config_merges_yaml_and_overrides(tmp_path):
    config_path = tmp_path / "iris.yaml"
    config_path.write_text("total_epochs: 12\ncollect_epsilon: 0.2\n", encoding="utf-8")

    composed = load_experiment_config(
        DemoObjectConfig(), config_path, ["collect_epsilon=0.05"]
    )

    assert composed["total_epochs"] == 12
    assert composed["collect_epsilon"] == 0.05
    assert composed["vocab_size"] == 512


def test_update_config_object_applies_composed_values():
    cfg = update_config_object(DemoObjectConfig(), {"total_epochs": 3})
    assert cfg.total_epochs == 3


def test_instantiate_dataclass_runs_post_init_after_composition(tmp_path):
    config_path = tmp_path / "demo.yaml"
    config_path.write_text("preset: small\nseed: 9\n", encoding="utf-8")

    cfg = instantiate_dataclass(DemoDataclassConfig, config_path)

    assert cfg.seed == 9
    assert cfg.preset == "small"
    assert cfg.width == 32
