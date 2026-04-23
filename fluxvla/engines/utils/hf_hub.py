from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]


def resolve_hf_local_path(model_name_or_path: str) -> str:
    """Resolve a local Hugging Face cache root to a concrete snapshot path."""
    path = Path(model_name_or_path).expanduser()
    if not path.exists() and path.is_absolute() is False:
        repo_relative_path = (REPO_ROOT / path).resolve()
        if repo_relative_path.exists():
            path = repo_relative_path

    if not path.exists() or not path.is_dir():
        return model_name_or_path

    refs_main = path / 'refs' / 'main'
    snapshots_dir = path / 'snapshots'
    if not snapshots_dir.is_dir():
        return model_name_or_path

    snapshot_name = None
    if refs_main.is_file():
        snapshot_name = refs_main.read_text(encoding='utf-8').strip()

    if snapshot_name:
        snapshot_path = snapshots_dir / snapshot_name
        if snapshot_path.is_dir():
            return str(snapshot_path)

    snapshot_dirs = sorted(child for child in snapshots_dir.iterdir()
                           if child.is_dir())
    if snapshot_dirs:
        return str(snapshot_dirs[0])

    return model_name_or_path
