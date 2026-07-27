"""Static release metadata invariants that should fail before publication."""

from __future__ import annotations

import re
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
_COMPATIBLE_RELEASE = re.compile(
    r"^(?P<name>[A-Za-z0-9_.-]+)~=(?P<version>[0-9]+\.[0-9]+\.[0-9]+)$"
)
_EXACT_PIN = re.compile(r"^(?P<name>[A-Za-z0-9_.-]+)==(?P<version>[0-9]+\.[0-9]+(?:\.[0-9]+)?)$")
_JOB_HEADING = re.compile(r"(?m)^  [a-z0-9-]+:\n")


def _normalized_name(value: str) -> str:
    return re.sub(r"[-_.]+", "-", value).lower()


def _version_tuple(value: str) -> tuple[int, ...]:
    return tuple(int(part) for part in value.split("."))


def _workflow_job(workflow: str, name: str) -> str:
    marker = f"  {name}:\n"
    start = workflow.index(marker)
    next_job = _JOB_HEADING.search(workflow, start + len(marker))
    end = next_job.start() if next_job is not None else len(workflow)
    return workflow[start:end]


def _workflow_job_permissions(workflow: str, name: str) -> set[str]:
    job = _workflow_job(workflow, name)
    match = re.search(
        r"(?m)^    permissions:\n(?P<body>(?:^      [^\n]+\n)+)",
        job,
    )
    assert match is not None, f"{name} must declare explicit permissions"
    return {line.strip() for line in match["body"].splitlines()}


def test_release_workflow_and_documentation_match_project_version():
    project = tomllib.loads((ROOT / "pyproject.toml").read_text())
    version = project["project"]["version"]
    workflow = (ROOT / ".github" / "workflows" / "release.yml").read_text()
    changelog = (ROOT / "CHANGELOG.md").read_text()
    checklist = (ROOT / "docs" / "release-checklist.md").read_text()

    assert f'default: "{version}"' in workflow
    assert f'test "$version" = "{version}"' in workflow
    assert f"## {version} — " in changelog
    assert f"release/v{version}" in checklist


def test_all_direct_dependencies_stay_on_tested_minor_lines():
    project = tomllib.loads((ROOT / "pyproject.toml").read_text())
    requirements = list(project["project"]["dependencies"])
    for extra in project["project"]["optional-dependencies"].values():
        requirements.extend(extra)

    declared: dict[str, tuple[int, ...]] = {}
    for requirement in requirements:
        match = _COMPATIBLE_RELEASE.fullmatch(requirement)
        assert match is not None, (
            f"{requirement!r} must use a three-component compatible-release "
            "bound (for example ~=5.13.0)"
        )
        declared[_normalized_name(match["name"])] = _version_tuple(match["version"])

    pins = {}
    for raw_line in (ROOT / "constraints.txt").read_text().splitlines():
        match = _EXACT_PIN.fullmatch(raw_line.strip())
        if match is not None:
            pins[_normalized_name(match["name"])] = _version_tuple(match["version"])

    for name, lower_bound in declared.items():
        assert name in pins, f"constraints.txt does not pin direct dependency {name}"
        pinned = pins[name]
        assert pinned >= lower_bound
        assert pinned[:2] == lower_bound[:2], (
            f"{name} pin {pinned} is outside declared tested minor line {lower_bound[:2]}"
        )


def test_release_publishes_the_smoke_tested_wheel_and_exact_constraints():
    workflow = (ROOT / ".github" / "workflows" / "release.yml").read_text()

    # One wheel build: the one subsequently installed and smoke-tested.
    assert workflow.count("python -m pip wheel . --no-deps --wheel-dir dist") == 1
    assert "actions/upload-artifact@" in workflow
    assert "actions/download-artifact@" in workflow
    assert "cp constraints.txt dist/constraints.txt" in workflow
    assert 'sha256sum "$wheel" dist/constraints.txt' in workflow
    assert "dist/*.whl dist/constraints.txt dist/SHA256SUMS" in workflow


def test_ci_and_release_enforce_ruff_formatting():
    for relative_path in (
        ".github/workflows/ci.yml",
        ".github/workflows/release.yml",
    ):
        workflow = (ROOT / relative_path).read_text()
        assert "ruff check src tests" in workflow
        assert "ruff format --check ." in workflow


def test_release_smoke_tests_immutable_images_before_publication():
    workflow = (ROOT / ".github" / "workflows" / "release.yml").read_text()

    smoke = 'tests/docker/smoke.sh "${IMAGE}:${TAG}" "$EXPECTED_CUDA"'
    assert workflow.count(smoke) == 1
    assert workflow.index("Smoke-test exact immutable image") < workflow.index(
        "promote-version-images:"
    )
    assert workflow.index("promote-version-images:") < workflow.index("release-metadata:")
    assert workflow.index("release-metadata:") < workflow.index("promote-rolling-images:")
    assert "needs: [verify-release, publish-images]" in workflow
    assert "needs: [verify-release, promote-version-images]" in workflow
    assert "needs: [verify-release, release-metadata]" in workflow


def test_release_publishes_only_after_exact_version_images_exist():
    workflow = (ROOT / ".github" / "workflows" / "release.yml").read_text()
    version_promotion = _workflow_job(workflow, "promote-version-images")
    rolling_promotion = _workflow_job(workflow, "promote-rolling-images")

    assert '--tag "${IMAGE}:${VERSION}-cu126"' in version_promotion
    assert '--tag "${IMAGE}:${VERSION}-cu130"' in version_promotion
    assert '--tag "${IMAGE}:${VERSION}"' in version_promotion
    assert "--tag" in rolling_promotion
    assert "${VERSION}" not in rolling_promotion


def test_release_promotes_the_verified_sha_if_master_advances_during_builds():
    workflow = (ROOT / ".github" / "workflows" / "release.yml").read_text()

    # Freshness is a dispatch-time gate. Once expensive image builds begin,
    # advancing master must not strand a published tag/release without aliases.
    assert workflow.count('test "$(git rev-parse HEAD)" = "$(git rev-parse origin/master)"') == 1
    for name in ("promote-version-images", "promote-rolling-images"):
        promotion = _workflow_job(workflow, name)
        assert "origin/master" not in promotion
        assert 'source="${IMAGE}:sha-${GITHUB_SHA}-cu126"' in promotion
        assert 'source="${IMAGE}:sha-${GITHUB_SHA}-cu130"' in promotion


def test_release_scopes_write_permissions_to_the_jobs_that_need_them():
    workflow = (ROOT / ".github" / "workflows" / "release.yml").read_text()
    workflow_header = workflow[: workflow.index("jobs:\n")]

    assert "permissions: {}\n" in workflow_header
    assert _workflow_job_permissions(workflow, "verify-release") == {"contents: read"}
    assert _workflow_job_permissions(workflow, "publish-images") == {
        "contents: read",
        "packages: write",
        "id-token: write",
        "attestations: write",
        "artifact-metadata: write",
    }
    assert _workflow_job_permissions(workflow, "promote-version-images") == {
        "contents: read",
        "packages: write",
    }
    assert _workflow_job_permissions(workflow, "release-metadata") == {
        "contents: write",
        "id-token: write",
        "attestations: write",
        "artifact-metadata: write",
    }
    assert _workflow_job_permissions(workflow, "promote-rolling-images") == {
        "contents: read",
        "packages: write",
    }


def test_release_acknowledges_unresolved_hardware_risk_without_claiming_a_soak():
    workflow = (ROOT / ".github" / "workflows" / "release.yml").read_text()

    assert "unresolved_slowdown_acknowledged:" in workflow
    assert 'test "$unresolved_slowdown_acknowledged" = "true"' in workflow
    assert "h100_soak_confirmed" not in workflow
    assert "h100_soak_evidence" not in workflow


def test_release_accepts_only_a_validated_versioned_release_branch_push():
    workflow = (ROOT / ".github" / "workflows" / "release.yml").read_text()

    assert 'branches:\n      - "release/v*"' in workflow
    assert 'release_ref="refs/heads/release/v${version}"' in workflow
    assert 'test "$GITHUB_REF" = "$release_ref"' in workflow
    assert 'requested_version="${GITHUB_REF#refs/heads/release/v}"' in workflow
    assert "unresolved_slowdown_acknowledged=true" in workflow
