"""Tests for the auto-generated Agent Skills Sphinx extension."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parent.parent
EXT_PATH = REPO_ROOT / "docs" / "_ext" / "skills_docs.py"


def _load_skills_docs():
    spec = importlib.util.spec_from_file_location("skills_docs", EXT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


skills_docs = _load_skills_docs()


def test_discover_repo_skills():
    skills = skills_docs.discover_skills(REPO_ROOT / "skills")
    names = [s.name for s in skills]
    assert "soprano-nmr" in names
    assert "soprano-spin-systems" in names
    assert "soprano-structures" in names
    for skill in skills:
        assert skill.title
        assert skill.description
        assert skill.summary
        assert skill.body


def test_generate_and_prune_skill_docs(tmp_path):
    skills_root = tmp_path / "skills"
    docs_root = tmp_path / "docs"

    # Create two synthetic skills (one with nested reference, one without references)
    alpha_dir = skills_root / "alpha-skill"
    (alpha_dir / "references" / "nested").mkdir(parents=True)
    (alpha_dir / "SKILL.md").write_text(
        "---\n"
        "name: alpha-skill\n"
        "description: Use for alpha tasks.\n"
        "---\n\n"
        "# Alpha Skill\n\n"
        "Handles alpha workflows cleanly.\n\n"
        "See [ref](references/nested/guide.md) and [beta](../beta-skill/SKILL.md).\n",
        encoding="utf-8",
    )
    (alpha_dir / "references" / "nested" / "guide.md").write_text(
        "# Nested Guide\n\n"
        "Back to [skill](../../SKILL.md) or [ext](https://example.com/SKILL.md).\n",
        encoding="utf-8",
    )

    beta_dir = skills_root / "beta-skill"
    beta_dir.mkdir(parents=True)
    (beta_dir / "SKILL.md").write_text(
        "No frontmatter or heading here.\n",
        encoding="utf-8",
    )

    repo_url = "https://github.com/CCP-NC/soprano/blob/master"
    generated = skills_docs.generate_skill_docs(skills_root, docs_root, repo_url)
    assert [s.name for s in generated] == ["alpha-skill", "beta-skill"]

    alpha_index = docs_root / "skills" / "alpha-skill" / "index.md"
    alpha_ref = (
        docs_root / "skills" / "alpha-skill" / "references" / "nested" / "guide.md"
    )
    beta_index = docs_root / "skills" / "beta-skill" / "index.md"
    summary_file = docs_root / "_generated" / "skills_summary.md"

    assert alpha_index.is_file()
    assert alpha_ref.is_file()
    assert beta_index.is_file()
    assert summary_file.is_file()

    alpha_text = alpha_index.read_text(encoding="utf-8")
    assert alpha_text.startswith("# Alpha Skill\n")
    assert "**Skill name:** `alpha-skill`" in alpha_text
    assert "Use for alpha tasks." in alpha_text
    assert "[beta](../beta-skill/index.md)" in alpha_text
    assert "[ref](references/nested/guide.md)" in alpha_text

    ref_text = alpha_ref.read_text(encoding="utf-8")
    assert ref_text.startswith("# Nested Guide\n")
    assert "[`alpha-skill`](../../index.md)" in ref_text
    assert "[skill](../../index.md)" in ref_text
    # External URLs must not be rewritten
    assert "[ext](https://example.com/SKILL.md)" in ref_text

    beta_text = beta_index.read_text(encoding="utf-8")
    assert beta_text.startswith("# beta-skill\n")

    summary_text = summary_file.read_text(encoding="utf-8")
    assert "[`alpha-skill`](skills/alpha-skill/index.md)" in summary_text
    assert "Handles alpha workflows cleanly." in summary_text
    assert "[`beta-skill`](skills/beta-skill/index.md)" in summary_text

    # Re-running with no changes should preserve mtime
    mtime_before = alpha_index.stat().st_mtime_ns
    skills_docs.generate_skill_docs(skills_root, docs_root, repo_url)
    assert alpha_index.stat().st_mtime_ns == mtime_before

    # Removing beta-skill should prune its generated page automatically
    (beta_dir / "SKILL.md").unlink()
    beta_dir.rmdir()
    skills_docs.generate_skill_docs(skills_root, docs_root, repo_url)
    assert not beta_index.exists()
    assert not (docs_root / "skills" / "beta-skill").exists()
    assert "beta-skill" not in summary_file.read_text(encoding="utf-8")
