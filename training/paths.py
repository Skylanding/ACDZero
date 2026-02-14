from datetime import datetime
from pathlib import Path
from typing import Tuple, Union


PathLike = Union[str, Path]


def create_timestamped_run_dirs(base_dir: PathLike, run_prefix: str) -> Tuple[Path, Path]:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = Path(base_dir) / f"{run_prefix}_{timestamp}"
    output_path.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = output_path / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    return output_path, checkpoint_dir


def resolve_training_data_dir(output_dir: PathLike) -> Path:
    output_path = Path(output_dir)

    if "unified_training" in str(output_path):
        for parent in [output_path.resolve()] + list(output_path.resolve().parents):
            if "unified_training" in parent.name:
                data_dir = parent / "training_data"
                data_dir.mkdir(parents=True, exist_ok=True)
                return data_dir

        parts = output_path.parts
        for i, part in enumerate(parts):
            if "unified_training" in part:
                data_dir = Path(*parts[: i + 1]) / "training_data"
                data_dir.mkdir(parents=True, exist_ok=True)
                return data_dir

    data_dir = output_path.parent / "training_data"
    data_dir.mkdir(parents=True, exist_ok=True)
    return data_dir

