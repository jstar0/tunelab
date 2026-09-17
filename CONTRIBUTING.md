# Contributing to TuneLab

Thanks for helping make experiments easier to inspect and reproduce.

Before opening a change:

1. Keep public behavior documented in the README and tests.
2. Preserve deterministic behavior for a fixed seed, objective, and search space.
3. Do not add client code, client data, credentials, or private examples.
4. Prefer a small, explicit change over a framework or abstraction.

Run the local checks:

```bash
uv run pytest
uv run --with ruff ruff check src tests
uv lock --check
uv build
```

If you add a strategy, document whether it is a faithful algorithm or an approximation, and add a test that checks candidate uniqueness and seeded repeatability.
