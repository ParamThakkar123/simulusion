import re
import tomllib
from pathlib import Path


def _dependency_names(dependencies):
    names = []
    for dependency in dependencies:
        if isinstance(dependency, str):
            names.append(re.split(r"[<>=!~; ]", dependency, maxsplit=1)[0])
        else:
            names.append(dependency["name"])
    return names


def test_jax_is_brax_optional_dependency_not_core_dependency():
    project = tomllib.loads(Path("pyproject.toml").read_text())["project"]

    assert "jax" not in _dependency_names(project["dependencies"])
    assert "jax" in _dependency_names(project["optional-dependencies"]["brax"])


def test_lockfile_keeps_jax_out_of_core_simulusion_dependencies():
    lock = tomllib.loads(Path("uv.lock").read_text())
    simulusion = next(
        package for package in lock["package"] if package["name"] == "simulusion"
    )

    assert "jax" not in _dependency_names(simulusion["dependencies"])
    assert "jax" in _dependency_names(simulusion["optional-dependencies"]["brax"])


def test_console_script_target_packages_are_included_in_setuptools_find():
    project = tomllib.loads(Path("pyproject.toml").read_text())
    scripts = project["project"]["scripts"]
    included_packages = set(
        project["tool"]["setuptools"]["packages"]["find"]["include"]
    )

    for target in scripts.values():
        module = target.split(":", maxsplit=1)[0]
        top_level_package = module.split(".", maxsplit=1)[0]
        assert top_level_package in included_packages


def test_click_is_core_dependency_for_cli():
    project = tomllib.loads(Path("pyproject.toml").read_text())["project"]

    assert "click" in _dependency_names(project["dependencies"])


def test_lockfile_keeps_click_in_core_simulusion_dependencies():
    lock = tomllib.loads(Path("uv.lock").read_text())
    simulusion = next(
        package for package in lock["package"] if package["name"] == "simulusion"
    )

    assert "click" in _dependency_names(simulusion["dependencies"])


def test_gymnasium_and_wandb_are_optional_not_core_dependencies():
    project = tomllib.loads(Path("pyproject.toml").read_text())["project"]

    core = _dependency_names(project["dependencies"])
    assert "gymnasium[box2d]" not in project["dependencies"]
    assert "wandb" not in core
    assert "gymnasium[box2d]" in _dependency_names(
        project["optional-dependencies"]["gym"]
    )
    assert "wandb" in _dependency_names(project["optional-dependencies"]["ml"])


def test_lockfile_keeps_gymnasium_and_wandb_out_of_core_simulusion_dependencies():
    lock = tomllib.loads(Path("uv.lock").read_text())
    simulusion = next(
        package for package in lock["package"] if package["name"] == "simulusion"
    )

    core = _dependency_names(simulusion["dependencies"])
    assert "gymnasium" not in core
    assert "wandb" not in core
    assert "gymnasium" in _dependency_names(simulusion["optional-dependencies"]["gym"])
    assert "wandb" in _dependency_names(simulusion["optional-dependencies"]["ml"])


def test_ml_agents_extra_matches_real_supported_sdk_series():
    project = tomllib.loads(Path("pyproject.toml").read_text())["project"]

    assert (
        "mlagents-envs>=0.28.0,<0.29.0" in project["optional-dependencies"]["ml-agents"]
    )
    # The extra must not reinstate a `protobuf<3.21` cap: the whole 3.x line is
    # unpatched for the advisories GitHub reports against `uv.lock`, and no
    # mlagents-envs release that lifts the cap is installable on Python >= 3.11.
    assert "protobuf>=5.29.6" in project["optional-dependencies"]["ml-agents"]


def test_no_extra_is_unsatisfiable_on_supported_pythons():
    project = tomllib.loads(Path("pyproject.toml").read_text())["project"]

    for name, requirements in project["optional-dependencies"].items():
        for requirement in requirements:
            assert "python_version < '3.11'" not in requirement, (name, requirement)
    assert "procgen" not in project["optional-dependencies"]


def test_repository_tools_are_not_installed_as_a_package():
    project = tomllib.loads(Path("pyproject.toml").read_text())
    included = project["tool"]["setuptools"]["packages"]["find"]["include"]

    assert "tools" not in included
    assert "tools.*" not in included


def test_dev_extra_installs_the_linter_the_makefile_runs():
    project = tomllib.loads(Path("pyproject.toml").read_text())["project"]

    assert "ruff" in _dependency_names(project["optional-dependencies"]["dev"])


def test_dmc_extra_installs_cleanly_on_every_supported_python():
    project = tomllib.loads(Path("pyproject.toml").read_text())["project"]
    dmc = project["optional-dependencies"]["dmc"]

    # dm-control pins `labmaze`, which has no CPython 3.13 wheel and builds with
    # Bazel, so an unconditional dm-control here fails the whole install on 3.13.
    dm_control = [item for item in dmc if item.startswith("dm-control")]
    assert dm_control == ["dm-control>=1.0.0; python_version < '3.13'"]
    assert any(
        item.startswith("labmaze-new") and "python_version >= '3.13'" in item
        for item in dmc
    )
    assert not any(item.split(";")[0].strip() == "labmaze" for item in dmc)


def test_uv_overrides_drop_labmaze_where_it_cannot_build():
    config = tomllib.loads(Path("pyproject.toml").read_text())["tool"]["uv"]

    assert "labmaze ; python_version < '3.13'" in config["override-dependencies"]


def test_dmc_uv_extra_keeps_dm_control_unconditional():
    project = tomllib.loads(Path("pyproject.toml").read_text())["project"]
    dmc_uv = project["optional-dependencies"]["dmc-uv"]

    assert "dm-control>=1.0.0" in dmc_uv
    assert not any(item.split(";")[0].strip() == "labmaze" for item in dmc_uv)


def test_gym_extra_does_not_pin_a_second_pygame_distribution():
    """gymnasium[box2d] >= 1.3 depends on pygame-ce, which installs the same
    `pygame` package directory as pygame. Requiring pygame as well put both in
    one environment and made env.render() crash the interpreter."""
    project = tomllib.loads(Path("pyproject.toml").read_text())["project"]

    names = _dependency_names(project["optional-dependencies"]["gym"])
    assert "pygame" not in names
    assert "pygame-ce" not in names
