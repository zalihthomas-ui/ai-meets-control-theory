# PyPI Packaging and Release Runbook

This document defines the release engineering procedures for building, verifying, and publishing the `aimct` distribution package to the Python Package Index (PyPI).

---

## 1. Distribution Architecture & Standards

- **Package Name**: `aimct`
- **Build Backend**: `setuptools.build_meta` (PEP 517 / PEP 621 compliant)
- **Distribution Artifacts**:
  - Pure Python Wheel: `dist/aimct-<version>-py3-none-any.whl`
  - Source Distribution: `dist/aimct-<version>.tar.gz`
- **Publishing Method**: GitHub Actions OpenID Connect (OIDC) **PyPI Trusted Publishing** (no long-lived API tokens or credentials).

---

## 2. Release Preparation Checklist

1. **Clean Test Suite**:
   ```bash
   pytest
   ```
2. **Version Bump**:
   - Update `version = "X.Y.Z"` in `pyproject.toml`.
   - Update `CHANGELOG.md` with release highlights conforming to Keep a Changelog.
3. **Artifact Hygiene**:
   - Verify `MANIFEST.in` excludes all binary PDFs (`docs/papers/*.pdf`), local experiment run artifacts (`experiments/**/runs/`), and temporary LaTeX build files.

---

## 3. Local Build & Validation

To test the package build locally without publishing:

```bash
# 1. Install build tools
pip install --upgrade build twine

# 2. Build distribution archives
python -m build

# 3. Validate package metadata and archive integrity
twine check dist/*

# 4. Inspect wheel contents to ensure no unwanted binary bloat
python -c "
import zipfile
with zipfile.ZipFile('dist/aimct-0.1.0-py3-none-any.whl') as z:
    for name in sorted(z.namelist()):
        print(name)
"
```

---

## 4. GitHub Actions Automated Publishing (OIDC)

Releases are published automatically upon tagging a commit on `main`:

```bash
git tag v0.1.0
git push origin v0.1.0
```

The GitHub Actions workflow (`.github/workflows/release.yml`):
1. Checks out the tagged commit.
2. Sets up Python and installs build dependencies.
3. Builds source and wheel distributions (`python -m build`).
4. Verifies distributions (`twine check dist/*`).
5. Uses `pypa/gh-action-pypi-publish@release/v1` with `id-token: write` permission to exchange a short-lived cryptographic OIDC token with PyPI.
