import importlib
import unittest

from model_paths import MODEL_PATHS


class SmokeTest(unittest.TestCase):
    def test_app_modules_import(self):
        for module_name in (
            "app",
            "classification.classify",
            "detection.detect",
            "segmentation.segment",
            "muscledetect.muscledetect",
        ):
            with self.subTest(module=module_name):
                importlib.import_module(module_name)

    def test_demo_model_files_exist(self):
        for model_key, model_path in MODEL_PATHS.items():
            with self.subTest(model=model_key):
                self.assertTrue(model_path.exists(), f"Missing {model_path}")


if __name__ == "__main__":
    unittest.main()
