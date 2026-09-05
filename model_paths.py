from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
MODELS_DIR = PROJECT_ROOT / "models"

MODEL_PATHS = {
    "blood_cell_detection": MODELS_DIR / "blood-cell-detection.pt",
    "lung_classification": MODELS_DIR / "lung-classification.pt",
    "ultrasound_segmentation": MODELS_DIR / "ultrasound-segmentation.pt",
    "fracture_classification": MODELS_DIR / "fracture-classification.pt",
}


def require_model(model_key, st=None):
    model_path = MODEL_PATHS[model_key]
    if model_path.exists():
        return str(model_path)

    message = (
        f"Missing model file: {model_path}. "
        "Restore it from the project release assets or retrain the corresponding model."
    )
    if st is not None:
        st.error(message)
        st.stop()
    raise FileNotFoundError(message)
