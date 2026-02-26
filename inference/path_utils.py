import os


INFERENCE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(INFERENCE_DIR)


def resolve_project_path(*parts: str) -> str:
    return os.path.join(PROJECT_ROOT, *parts)


def default_highlight_output_dir() -> str:
    return resolve_project_path("inference", "outputs", "highlights")


def default_model_registry_path() -> str:
    return resolve_project_path("inference", "config", "models.json")
