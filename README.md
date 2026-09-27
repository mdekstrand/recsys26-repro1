# RecSys Reproducibility Tutorial Example 1

This is the first example for the RecSys 2026 Reproducibility tutorial.

## Installation

[uv]: https://docs.astral.sh/uv/

This project uses [uv][] to install its dependencies.  First install `uv`:

- on macOS, `brew install uv`
- on Windows, `winget install astral-sh.uv`
- on Linux, follow [install instructions](https://docs.astral.sh/uv/getting-started/installation/)

Once `uv` is installed, you can install and reproduce this project with:

```console
uv sync
uv run dvc repro
```
