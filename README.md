# sltop: A top-like queue viewer for Slurm

A slightly more aesthetically pleasing version of `squeue`, with node usage
bars and a scrollable, refreshing job list.

## Installation

```bash
uv tool install sltop --from git+https://github.com/thetianshuhuang/sltop.git
```

## Usage

```bash
sltop              # Interactive view, refreshing once per second.
sltop --static     # Print the job list once and exit.
sltop -p dev       # Only show jobs and nodes in the `dev` partition.
```

| Key | Action |
| --- | --- |
| `i` | Show details for the selected job |
| `↑`/`↓`, `j`/`k` | Move the selection |
| `PgUp`/`PgDn` | Page through the list |
| `g`/`G` | Jump to the top/bottom |
| `q` | Quit |

## Development

```bash
uv sync                    # Install, including the dev dependencies.
uv run pre-commit install  # Lint, type check and test on every commit.
uv run pytest tests        # Or run the tests directly.
```
