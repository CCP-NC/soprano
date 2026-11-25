"""Sphinx extension to auto-generate docs from the repository's ``skills/`` directory.

On every Sphinx/Jupyter Book build, this extension:

1. Scans ``<repo>/skills/*/SKILL.md`` and any ``references/**/*.md`` files.
2. Renders up-to-date MyST Markdown pages under ``docs/skills/<skill>/`` and a
   summary table under ``docs/_generated/skills_summary.md``.
3. Registers the generated skill pages (and their reference subpages) under the
   ``agent-skills`` entry in ``sphinx-external-toc``'s ``external_site_map``.

Because all pages are generated at build time directly from ``skills/``, the
documentation stays in sync automatically whenever skills are added, edited,
or removed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
import re
from typing import Any

try:
    import yaml
except ImportError:  # pragma: no cover - PyYAML is always installed in docs builds
    yaml = None

_FRONTMATTER_RE = re.compile(r"\A---\s*\n(.*?)\n---\s*(?:\n|\Z)", re.DOTALL)
_H1_RE = re.compile(r"^#\s+(.+?)\s*$")
_SKILL_MD_LINK_RE = re.compile(
    r"(\[[^\]]*\]\()(?![a-zA-Z][a-zA-Z0-9+.-]*://)([^)#]*?)(SKILL\.md)(#[^)]*)?(\))"
)


@dataclass(frozen=True)
class ReferenceEntry:
    """A reference Markdown document belonging to a skill."""

    source_path: Path
    # Path relative to skills/<skill>/references/, e.g. "cli-reference.md"
    rel_posix: str
    title: str
    body: str

    @property
    def doc_rel_posix(self) -> str:
        """Path relative to ``docs/`` without extension."""
        stem = PurePosixPath(self.rel_posix).with_suffix("").as_posix()
        return f"references/{stem}"


@dataclass(frozen=True)
class SkillEntry:
    """A parsed Agent Skill discovered under ``skills/<dir_name>/``."""

    dir_name: str
    name: str
    description: str
    title: str
    summary: str
    body: str
    source_path: Path
    references: list[ReferenceEntry] = field(default_factory=list)


def _parse_simple_yaml(raw_yaml: str) -> dict[str, str]:
    """Minimal fallback parser for flat ``key: value`` frontmatter without PyYAML."""
    result: dict[str, str] = {}
    current_key: str | None = None
    for line in raw_yaml.splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        if line[0].isspace() and current_key is not None:
            result[current_key] = f"{result[current_key]} {line.strip()}".strip()
        elif ":" in line:
            key, _, val = line.partition(":")
            current_key = key.strip()
            result[current_key] = val.strip().strip("'\"")
    return result


def parse_frontmatter(text: str) -> tuple[dict[str, Any], str]:
    """Split optional YAML frontmatter from a Markdown document."""
    match = _FRONTMATTER_RE.match(text)
    if not match:
        return {}, text
    raw_yaml = match.group(1)
    if yaml is not None:
        parsed = yaml.safe_load(raw_yaml)
        metadata = parsed if isinstance(parsed, dict) else {}
    else:
        metadata = _parse_simple_yaml(raw_yaml)
    body = text[match.end() :]
    return metadata, body


def extract_title_and_body(markdown: str, fallback_title: str) -> tuple[str, str]:
    """Extract the first H1 heading and return ``(title, remaining_body)``."""
    lines = markdown.splitlines()
    for idx, line in enumerate(lines):
        stripped = line.strip()
        if not stripped:
            continue
        match = _H1_RE.match(stripped)
        if match:
            title = match.group(1).strip()
            remaining = "\n".join(lines[idx + 1 :]).lstrip("\n")
            return title, remaining
        # Non-empty content before any H1: synthesize title and keep full body.
        break
    return fallback_title, markdown.lstrip("\n")


def _first_sentence(text: str) -> str:
    """Return the first sentence of ``text`` (or the full string if no break)."""
    cleaned = " ".join(text.split())
    if not cleaned:
        return ""
    match = re.match(r"^(.*?[.!?])(?:\s+|$)", cleaned)
    return match.group(1) if match else cleaned


def extract_intro_summary(body: str, fallback_description: str) -> str:
    """Extract the opening prose paragraph from ``body`` for the summary table."""
    paragraph_lines: list[str] = []
    in_fence = False

    for line in body.splitlines():
        stripped = line.strip()
        if stripped.startswith(("```", "~~~")):
            in_fence = not in_fence
            if paragraph_lines:
                break
            continue
        if in_fence:
            continue
        if not stripped:
            if paragraph_lines:
                break
            continue
        if stripped.startswith(("#", ">", "-", "*", "|", ":::")):
            if paragraph_lines:
                break
            if stripped.startswith("##"):
                break
            continue
        paragraph_lines.append(stripped)

    summary = (
        " ".join(paragraph_lines)
        if paragraph_lines
        else _first_sentence(fallback_description)
    )
    return summary.replace("|", r"\|")


def rewrite_skill_links(text: str) -> str:
    """Rewrite relative Markdown links pointing to ``SKILL.md`` to ``index.md``."""
    return _SKILL_MD_LINK_RE.sub(
        lambda m: f"{m.group(1)}{m.group(2)}index.md{m.group(4) or ''}{m.group(5)}",
        text,
    )


def discover_skills(skills_root: Path) -> list[SkillEntry]:
    """Discover and parse all skills under ``skills_root``."""
    if not skills_root.is_dir():
        return []

    skills: list[SkillEntry] = []
    for skill_dir in sorted(p for p in skills_root.iterdir() if p.is_dir()):
        skill_md = skill_dir / "SKILL.md"
        if not skill_md.is_file():
            continue

        raw_text = skill_md.read_text(encoding="utf-8")
        meta, without_fm = parse_frontmatter(raw_text)
        name = str(meta.get("name") or skill_dir.name).strip()
        description = " ".join(str(meta.get("description") or "").split())
        title, body = extract_title_and_body(without_fm, fallback_title=name)
        summary = extract_intro_summary(body, fallback_description=description)

        references: list[ReferenceEntry] = []
        refs_dir = skill_dir / "references"
        if refs_dir.is_dir():
            for ref_file in sorted(refs_dir.rglob("*.md")):
                if not ref_file.is_file():
                    continue
                rel_posix = ref_file.relative_to(refs_dir).as_posix()
                ref_raw = ref_file.read_text(encoding="utf-8")
                _, ref_without_fm = parse_frontmatter(ref_raw)
                fallback = ref_file.stem.replace("-", " ").replace("_", " ").title()
                ref_title, ref_body = extract_title_and_body(
                    ref_without_fm, fallback_title=fallback
                )
                references.append(
                    ReferenceEntry(
                        source_path=ref_file,
                        rel_posix=rel_posix,
                        title=ref_title,
                        body=ref_body,
                    )
                )

        skills.append(
            SkillEntry(
                dir_name=skill_dir.name,
                name=name,
                description=description,
                title=title,
                summary=summary,
                body=body,
                source_path=skill_md,
                references=references,
            )
        )

    return skills


def render_skill_page(skill: SkillEntry, repo_blob_url: str) -> str:
    """Render the MyST Markdown page for a single skill."""
    source_rel = f"skills/{skill.dir_name}/SKILL.md"
    source_url = f"{repo_blob_url.rstrip('/')}/{source_rel}"

    meta_lines = [f"- **Skill name:** `{skill.name}`"]
    if skill.description:
        meta_lines.append(f"- **When agents load it:** {skill.description}")
    meta_lines.append(f"- **Source:** [`{source_rel}`]({source_url})")

    admonition = "\n".join(
        [
            ":::{admonition} Skill metadata",
            ":class: note",
            "",
            *meta_lines,
            ":::",
        ]
    )

    body = rewrite_skill_links(skill.body).rstrip()
    return f"# {skill.title}\n\n{admonition}\n\n{body}\n"


def render_reference_page(
    skill: SkillEntry, ref: ReferenceEntry, repo_blob_url: str
) -> str:
    """Render the MyST Markdown page for a skill's reference document."""
    source_rel = f"skills/{skill.dir_name}/references/{ref.rel_posix}"
    source_url = f"{repo_blob_url.rstrip('/')}/{source_rel}"

    # Compute relative path from references/<nested>/file.md back to ../index.md
    depth = len(PurePosixPath(ref.rel_posix).parts)
    parent_link = ("../" * depth) + "index.md"

    note = "\n".join(
        [
            ":::{note}",
            f"Reference document for the [`{skill.name}`]({parent_link}) skill. "
            f"Source: [`{source_rel}`]({source_url}).",
            ":::",
        ]
    )

    body = rewrite_skill_links(ref.body).rstrip()
    return f"# {ref.title}\n\n{note}\n\n{body}\n"


def render_summary_table(skills: list[SkillEntry]) -> str:
    """Render a Markdown summary table of all discovered skills."""
    if not skills:
        return "_No Agent Skills were found in `skills/`._\n"

    lines = [
        "| Skill | Overview |",
        "|---|---|",
    ]
    for skill in skills:
        link = f"[`{skill.name}`](skills/{skill.dir_name}/index.md)"
        lines.append(f"| {link} | {skill.summary} |")
    return "\n".join(lines) + "\n"


def _write_if_changed(path: Path, content: str) -> bool:
    """Write ``content`` to ``path`` only if the file does not exist or differs."""
    if path.exists() and path.read_text(encoding="utf-8") == content:
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return True


def _prune_stale_files(output_dir: Path, expected_files: set[Path]) -> None:
    """Remove stale generated files and empty directories in ``output_dir``."""
    if not output_dir.is_dir():
        return
    for existing in list(output_dir.rglob("*")):
        if existing.is_file() and existing not in expected_files:
            existing.unlink()
    # Remove empty directories bottom-up
    for existing in sorted(output_dir.rglob("*"), reverse=True):
        if existing.is_dir() and not any(existing.iterdir()):
            existing.rmdir()


def generate_skill_docs(
    skills_root: Path, docs_root: Path, repo_blob_url: str
) -> list[SkillEntry]:
    """Generate skill documentation pages and summary snippet from ``skills_root``."""
    skills = discover_skills(skills_root)
    skills_out = docs_root / "skills"
    generated_dir = docs_root / "_generated"

    expected_files: set[Path] = set()
    for skill in skills:
        skill_page = skills_out / skill.dir_name / "index.md"
        expected_files.add(skill_page)
        _write_if_changed(skill_page, render_skill_page(skill, repo_blob_url))

        for ref in skill.references:
            ref_page = skills_out / skill.dir_name / "references" / ref.rel_posix
            expected_files.add(ref_page)
            _write_if_changed(
                ref_page, render_reference_page(skill, ref, repo_blob_url)
            )

    _prune_stale_files(skills_out, expected_files)

    summary_path = generated_dir / "skills_summary.md"
    _write_if_changed(summary_path, render_summary_table(skills))

    return skills


def _generate_on_config_inited(app: Any, config: Any) -> None:
    """Sphinx ``config-inited`` hook (before external-toc): generate Markdown pages."""
    from sphinx.util import logging

    logger = logging.getLogger(__name__)
    docs_root = Path(app.srcdir)
    skills_root = (
        Path(config.skills_docs_source_dir)
        if config.skills_docs_source_dir
        else docs_root.parent / "skills"
    )
    if not skills_root.is_dir():
        logger.warning("[skills_docs] Skills directory not found: %s", skills_root)
        app._soprano_skills = []
        return

    skills = generate_skill_docs(
        skills_root=skills_root,
        docs_root=docs_root,
        repo_blob_url=config.skills_docs_repo_blob_url,
    )
    app._soprano_skills = skills
    logger.info(
        "[skills_docs] Generated documentation for %d skill(s) from %s",
        len(skills),
        skills_root,
    )


def _register_toc_on_config_inited(app: Any, config: Any) -> None:
    """Sphinx ``config-inited`` hook: attach generated skill pages to the TOC."""
    from sphinx_external_toc.api import Document, FileItem, TocTree

    skills: list[SkillEntry] = getattr(app, "_soprano_skills", [])
    if not skills:
        return

    site_map = getattr(config, "external_site_map", None)
    parent_docname = config.skills_docs_parent_doc
    if site_map is None or parent_docname not in site_map:
        return

    parent_doc = site_map[parent_docname]
    skill_items: list[FileItem] = []
    generated_paths: set[str] = set()

    for skill in skills:
        skill_docname = f"skills/{skill.dir_name}/index"
        generated_paths.add(f"{skill_docname}.md")

        ref_items: list[FileItem] = []
        for ref in skill.references:
            ref_docname = f"skills/{skill.dir_name}/{ref.doc_rel_posix}"
            generated_paths.add(f"{ref_docname}.md")
            site_map[ref_docname] = Document(docname=ref_docname)
            ref_items.append(FileItem(ref_docname))

        subtrees = [TocTree(items=ref_items)] if ref_items else []
        site_map[skill_docname] = Document(
            docname=skill_docname,
            title=skill.name,
            subtrees=subtrees,
        )
        skill_items.append(FileItem(skill_docname))

    if skill_items:
        parent_doc.subtrees.append(TocTree(items=skill_items))

    # If external_toc_exclude_missing was enabled, ensure generated skill docs
    # are not left in exclude_patterns.
    if config.exclude_patterns:
        config.exclude_patterns = [
            p for p in config.exclude_patterns if p not in generated_paths
        ]


def setup(app: Any) -> dict[str, Any]:
    """Register the ``skills_docs`` Sphinx extension."""
    app.add_config_value("skills_docs_source_dir", None, "env")
    app.add_config_value("skills_docs_parent_doc", "agent-skills", "env")
    app.add_config_value(
        "skills_docs_repo_blob_url",
        "https://github.com/CCP-NC/soprano/blob/master",
        "env",
    )
    # Generate Markdown before sphinx_external_toc runs (priority 900)
    app.connect("config-inited", _generate_on_config_inited, priority=500)
    # Attach generated pages to external_site_map right after sphinx_external_toc
    app.connect("config-inited", _register_toc_on_config_inited, priority=950)
    return {
        "version": "0.1.0",
        "parallel_read_safe": True,
        "parallel_write_safe": True,
    }
