from pathlib import Path


def test_publish_workflow_updates_dynamic_version_source():
    workflow = Path(".github/workflows/publish-pypi.yml").read_text(encoding="utf-8")

    assert 'Path("simulusion/_version.py")' in workflow
    assert "Could not find __version__ in simulusion/_version.py" in workflow
    assert "git add simulusion/_version.py" in workflow
    assert "top-level project version in pyproject.toml" not in workflow
