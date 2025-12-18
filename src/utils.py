from __future__ import annotations

from pathlib import Path


def project_root() -> Path:
	# Devuelve la ruta raíz del repo
	return Path(__file__).resolve().parents[1]


def data_dir() -> Path:
	return project_root() / "data"


def models_dir() -> Path:
	return project_root() / "models"


def ensure_dir(path: Path) -> Path:
	path.mkdir(parents=True, exist_ok=True)
	return path
