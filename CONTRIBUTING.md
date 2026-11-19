# Contributing

Thanks for your interest in improving NEXUS. Focus changes on an existing behavior or a clearly described issue, and keep documentation aligned with the code.

## Set up

The project currently requires CPython 3.15 (`>=3.15,<3.16`). Install the development environment with:

```bash
uv sync --all-extras
```

## Before opening a change

```bash
uv run pytest
uv run ruff check .
uv run mypy nexus
```

Add or update tests when changing application behavior. For documentation-only changes, check links, commands, spelling, and rendered Markdown. Avoid adding generated reports, local data, credentials, or trained weights.

## Code guidelines

- Keep changes focused and explain user-visible behavior.
- Prefer clear names, small functions, and type annotations where useful.
- Preserve deterministic behavior where practical.
- Avoid new dependencies unless the change needs them.
- Do not claim a feature, test count, or performance result without checking the current implementation.

## Pull requests

Describe the problem, the change, and how it was verified. Include relevant test results and note any configuration or migration steps. Use a short, imperative commit subject, such as `docs: update setup instructions`.

## Security issues

Do not report vulnerabilities in a public issue. Follow the private reporting instructions in [SECURITY.md](SECURITY.md).

## License

By submitting a contribution, you agree it is distributed under the project’s [MIT License](LICENSE).
