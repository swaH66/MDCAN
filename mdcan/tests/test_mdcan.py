import gc
import os
import tempfile
import unittest
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from models import DCHA, ESA, MDCAN, MSCA, MSFB, create_mdcan
from train import SkinLesionFolder, build_transforms
from utils import load_checkpoint


class MDCANTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        requested_device = os.getenv("MDCAN_TEST_DEVICE", "cpu")
        if requested_device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available.")
        cls.device = torch.device(requested_device)
        cls.num_classes = int(os.getenv("MDCAN_TEST_NUM_CLASSES", "3"))
        cls.data_path = os.getenv("MDCAN_TEST_DATA_PATH")
        cls.checkpoint = os.getenv("MDCAN_TEST_CHECKPOINT")
        cls.max_samples = int(os.getenv("MDCAN_TEST_MAX_SAMPLES", "8"))

    def tearDown(self):
        gc.collect()
        if self.device.type == "cuda":
            torch.cuda.empty_cache()

    def test_public_model_and_module_names(self):
        model = create_mdcan(pretrained=False, num_classes=self.num_classes)
        self.assertIsInstance(model, MDCAN)
        self.assertIsInstance(model.global_path[1], MSCA)
        self.assertIsInstance(model.global_path[1].dcha, DCHA)
        self.assertIsInstance(model.global_path[1].esa, ESA)
        self.assertIsInstance(model.global_path[1].msfb, MSFB)
        self.assertEqual(model.classifier[-1].out_features, self.num_classes)

    def test_dual_path_msca_parameters_are_independent(self):
        model = create_mdcan(pretrained=False, num_classes=self.num_classes)
        self.assertIsNot(model.global_path[1], model.local_path[1])
        self.assertIsNot(model.global_path[1].dcha, model.local_path[1].dcha)

    def test_forward_output_shape(self):
        model = create_mdcan(pretrained=False, num_classes=self.num_classes).to(self.device)
        model.eval()
        sample = torch.randn(1, 3, 224, 224, device=self.device)
        with torch.inference_mode():
            output = model(sample)
        self.assertEqual(tuple(output.shape), (1, self.num_classes))

    def test_training_checkpoint_loads_strictly(self):
        source = create_mdcan(pretrained=False, num_classes=self.num_classes)
        target = create_mdcan(pretrained=False, num_classes=self.num_classes)
        with tempfile.TemporaryDirectory() as temporary_directory:
            checkpoint_path = Path(temporary_directory) / "mdcan.pth"
            torch.save({"model": source.state_dict(), "epoch": 7}, checkpoint_path)
            metadata = load_checkpoint(target, checkpoint_path, strict=True)
        self.assertEqual(metadata["epoch"], 7)
        self.assertEqual(metadata["missing_keys"], [])
        self.assertEqual(metadata["unexpected_keys"], [])

    def test_dataparallel_prefix_is_removed(self):
        source = create_mdcan(pretrained=False, num_classes=self.num_classes)
        prefixed = {f"module.{key}": value for key, value in source.state_dict().items()}
        target = create_mdcan(pretrained=False, num_classes=self.num_classes)
        with tempfile.TemporaryDirectory() as temporary_directory:
            checkpoint_path = Path(temporary_directory) / "parallel.pth"
            torch.save({"state_dict": prefixed}, checkpoint_path)
            metadata = load_checkpoint(target, checkpoint_path, strict=True)
        self.assertEqual(metadata["missing_keys"], [])
        self.assertEqual(metadata["unexpected_keys"], [])

    def test_external_checkpoint_loads_strictly(self):
        if not self.checkpoint:
            self.skipTest("No --checkpoint was provided.")
        checkpoint_path = Path(self.checkpoint)
        self.assertTrue(checkpoint_path.is_file(), f"Checkpoint not found: {checkpoint_path}")
        model = create_mdcan(pretrained=False, num_classes=self.num_classes)
        metadata = load_checkpoint(model, checkpoint_path, device="cpu", strict=True)
        self.assertEqual(metadata["missing_keys"], [])
        self.assertEqual(metadata["unexpected_keys"], [])

    def test_custom_dataset_pipeline(self):
        if not self.data_path:
            self.skipTest("No --data-path was provided.")

        data_path = Path(self.data_path)
        split_path = data_path / "test" if (data_path / "test").is_dir() else data_path
        _, evaluation_transform = build_transforms()
        dataset = SkinLesionFolder(split_path, evaluation_transform)
        self.assertEqual(
            len(dataset.classes),
            self.num_classes,
            f"Expected {self.num_classes} classes, found {dataset.classes}",
        )

        sample_count = min(len(dataset), self.max_samples)
        self.assertGreater(sample_count, 0)
        loader = DataLoader(
            torch.utils.data.Subset(dataset, range(sample_count)),
            batch_size=min(4, sample_count),
            shuffle=False,
            num_workers=0,
        )
        model = create_mdcan(pretrained=False, num_classes=self.num_classes).to(self.device)
        if self.checkpoint:
            load_checkpoint(model, self.checkpoint, device=self.device, strict=True)
        model.eval()

        processed = 0
        with torch.inference_mode():
            for images, labels in loader:
                outputs = model(images.to(self.device))
                self.assertEqual(outputs.ndim, 2)
                self.assertEqual(outputs.shape[0], images.shape[0])
                self.assertEqual(outputs.shape[1], self.num_classes)
                self.assertTrue(torch.isfinite(outputs).all().item())
                self.assertTrue(((labels >= 0) & (labels < self.num_classes)).all().item())
                processed += images.shape[0]
        self.assertEqual(processed, sample_count)


if __name__ == "__main__":
    unittest.main()
