# Documentation Portal & Developer Guide

This hosted documentation portal is built with **MkDocs**, the **Material for MkDocs** theme, **mkdocstrings** (for auto-generated Python API docs), and **mkdocs-jupyter** (for rendering Jupyter notebooks).

---

## 1. Local Preview & Development

To preview the documentation portal locally with live reload:

```bash
# 1. Install documentation dependencies
pip install -e ".[docs]"

# 2. Launch the local development server
mkdocs serve
```

Open your browser at `http://127.0.0.1:8000` to browse the portal with live hot-reloading on every file edit.

To build the static site locally:

```bash
mkdocs build --strict
```

The compiled static HTML/JS/CSS site will be written to `site/` (which is git-ignored).

---

## 2. Automated GitHub Pages Deployment

The documentation portal is automatically built and deployed to **GitHub Pages** via the GitHub Actions workflow `.github/workflows/docs.yml` on every push to the `main` branch.

### Deployment Workflow Architecture

1. **Trigger:** Push to `main` modifying `docs/`, `src/`, `notebooks/`, `mkdocs.yml`, or `pyproject.toml`.
2. **Environment:** Ubuntu Latest with Python 3.12.
3. **Build:** Installs `aimct` with `[docs,ml,xcheck]` extras and builds the site using `mkdocs gh-deploy --force`.
4. **Publish:** Publishes the static site to the `gh-pages` branch hosted at `https://zalihthomas-ui.github.io/ai-meets-control-theory/`.

---

## 3. Structure & Source of Truth

The documentation portal enforces a **single source of truth** architecture:
- Core engineering guides (`docs/DECISION-GUIDE.md`, `docs/RESULTS.md`, `docs/USAGE.md`, `docs/GETTING-STARTED.md`, `docs/VISUALIZATION.md`, `docs/DEV_PREVIEW.md`) are directly rendered in the navigation tree without duplicating or forking content.
- Interactive tutorials (`notebooks/01_tour.ipynb`) are rendered natively via `mkdocs-jupyter`.
- API Reference pages (`docs/api/*.md`) dynamically introspect and render docstrings directly from `src/aimct/` via `mkdocstrings[python]`.
