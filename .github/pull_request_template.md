## Description
<!-- Provide a clear, concise summary of the changes introduced in this PR. -->

## Type of Change
- [ ] 🐛 Bug fix (non-breaking fix for an existing issue)
- [ ] ✨ New feature (non-breaking addition of system, controller, estimator, or tool)
- [ ] 📈 Performance optimization / Numerical precision improvement
- [ ] 📚 Documentation update (concepts, tutorials, docstrings, API reference)
- [ ] 🔬 New empirical benchmark / experiment (`experiments/NN_...`)
- [ ] ⚠️ Breaking change (violates SemVer 2.0.0 public API contract - requires discussion)

## Validation Checklist
- [ ] All unit tests pass locally (`pytest tests/ -q`)
- [ ] No regression on slow test suite (`pytest -m "not slow"`)
- [ ] Code follows project formatting and style conventions (`ruff check src/ tests/`)
- [ ] Type hints verified (`mypy src/aimct`)
- [ ] Documentation builds strictly with zero warnings (`mkdocs build --strict`)
- [ ] Added or updated unit tests covering the new functionality
- [ ] If a new experiment is added: includes `config.yaml`, `run.py`, `table.md`, `table.csv`, and `figure.png`
- [ ] If changing public API: documented in docstrings and `docs/STABILITY.md` verified
