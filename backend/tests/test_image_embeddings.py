"""Deterministic index integrity, incremental migration and pixel-ranking tests."""
import tempfile
import unittest
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image

from app.core.config import Settings
from app.services.image_embeddings import DIMENSION, ImageEmbeddingIndex, normalized, read_image


class FakeSettings(Settings):
    test_root: Path

    @property
    def project_root(self):
        return self.test_root


class PixelEncoder:
    """No API/model downloads: deterministic orthogonal pixel/query features."""
    def __init__(self, _):
        self.encoded = []
        self.queries = []

    def images(self, paths):
        self.encoded.extend(paths)
        vectors = np.zeros((len(paths), DIMENSION), dtype=np.float32)
        for index, path in enumerate(paths):
            r, g, b = read_image(path).getpixel((0, 0))
            vectors[index, 0 if r > b else 1] = 1
        return vectors

    def text(self, query):
        self.queries.append(query)
        vector = np.zeros(DIMENSION, dtype=np.float32)
        vector[0 if "red" in query else 1] = 1
        return vector


class ImageIndexTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.settings = FakeSettings(test_root=self.root, image_embeddings_enabled=True)
        self.folder = self.settings.data_dir / "raw/traffic_signs"
        self.folder.mkdir(parents=True)
        self.red = self.folder / "red.png"
        self.blue = self.folder / "blue.png"
        Image.new("RGB", (16, 16), "red").save(self.red)
        Image.new("RGB", (16, 16), "blue").save(self.blue)
        self.encoder = PixelEncoder(None)
        self.index = ImageEmbeddingIndex(self.settings, lambda _: self.encoder)

    def test_images_are_embedded_and_text_uses_the_same_space(self):
        manifest = self.index.build()
        self.assertEqual(len(manifest["assets"]), 2)
        scores = self.index.scores("red sign", {"red-record": [self.red], "blue-record": [self.blue]})
        self.assertGreater(scores["red-record"], scores["blue-record"])
        self.assertAlmostEqual(scores["red-record"], 1)
        self.assertEqual(self.encoder.queries, ["red sign"])
        self.index.scores("red sign", {"red-record": [self.red]})
        self.assertEqual(len(self.encoder.queries), 1)

    def test_migration_deduplicates_pixels_and_reuses_unchanged_vectors(self):
        duplicate = self.settings.data_dir / "raw/images/duplicate.png"
        duplicate.parent.mkdir(parents=True)
        duplicate.write_bytes(self.red.read_bytes())
        initial = self.index.build()
        self.assertEqual(initial["file_count"], 3)
        self.assertEqual(len(initial["assets"]), 2)
        previous_calls = len(self.encoder.encoded)
        repeated = self.index.build()
        self.assertEqual(repeated["reused_vectors"], 2)
        self.assertEqual(len(self.encoder.encoded), previous_calls)
        Image.new("RGB", (16, 16), "green").save(self.folder / "new.png")
        updated = self.index.build()
        self.assertEqual(updated["file_count"], 4)
        self.assertEqual(len(self.encoder.encoded), previous_calls + 1)

    def test_missing_changed_and_outside_images_cannot_use_stale_associations(self):
        self.index.build()
        outside = self.root / "outside.png"
        outside.write_bytes(self.red.read_bytes())
        self.red.unlink()
        Image.new("RGB", (16, 16), "green").save(self.blue)
        self.assertEqual(self.index.scores("red", {"deleted": [self.red], "changed": [self.blue], "outside": [outside]}), {})

    def test_reload_validates_model_dimensions_and_hash_associations(self):
        self.index.build()
        reloaded = ImageEmbeddingIndex(self.settings, lambda _: self.encoder)
        self.assertTrue(reloaded.status()["indexed"])
        with np.load(self.settings.image_index_path, allow_pickle=False) as saved:
            manifest = saved["manifest"]
        with self.settings.image_index_path.open("wb") as handle:
            np.savez(handle, manifest=manifest, vectors=np.ones((2, 384), dtype=np.float32))
        corrupt = ImageEmbeddingIndex(self.settings, lambda _: self.encoder)
        self.assertFalse(corrupt.status()["indexed"])
        self.assertEqual(corrupt.scores("red", {"red": [self.red]}), {})

    def test_bad_image_is_skipped_and_valid_images_remain_searchable(self):
        (self.folder / "bad.png").write_bytes(b"invalid image bytes")
        manifest = self.index.build()
        self.assertEqual(len(manifest["assets"]), 2)
        self.assertEqual(len(manifest["skipped"]), 1)
        self.assertTrue(self.index.scores("red", {"red": [self.red]}))

    def test_encoder_failure_uses_safe_fallback_without_overwriting_index(self):
        self.index.build()
        before = self.settings.image_index_path.read_bytes()
        unavailable = ImageEmbeddingIndex(self.settings, lambda _: (_ for _ in ()).throw(RuntimeError("model unavailable")))
        self.assertEqual(unavailable.scores("red", {"red": [self.red]}), {})
        self.assertEqual(self.settings.image_index_path.read_bytes(), before)
        self.assertEqual(unavailable.error, "RuntimeError")

    def test_zero_nonfinite_and_incompatible_vectors_are_rejected(self):
        for vectors in (np.zeros((1, DIMENSION)), np.ones((1, 384)), np.full((1, DIMENSION), np.nan)):
            with self.subTest(shape=vectors.shape):
                with self.assertRaises(ValueError):
                    normalized(vectors)

    def test_running_index_observes_atomic_cache_updates(self):
        self.index.build()
        running = ImageEmbeddingIndex(self.settings, lambda _: self.encoder)
        self.assertEqual(running.status()["indexed_files"], 2)
        added = self.folder / "added.png"
        Image.new("RGB", (16, 16), "green").save(added)
        self.index.build()
        self.assertEqual(running.status()["indexed_files"], 3)
        self.assertIn("new-record", running.scores("blue", {"new-record": [added]}))

    def test_health_snapshot_does_not_wait_for_encoder_inference(self):
        self.index.build()
        entered, release = threading.Event(), threading.Event()
        original = self.encoder.text

        def delayed(query):
            entered.set()
            release.wait(3)
            return original(query)

        self.encoder.text = delayed
        with ThreadPoolExecutor(max_workers=2) as pool:
            scoring = pool.submit(self.index.scores, "red", {"red": [self.red]})
            self.assertTrue(entered.wait(2))
            try:
                status = pool.submit(self.index.status).result(timeout=1)
                self.assertTrue(status["indexed"])
            finally:
                release.set()
            self.assertTrue(scoring.result(timeout=2))

    def test_svg_preserves_full_symbol_and_rejects_external_resources(self):
        svg = self.folder / "safe.svg"
        svg.write_text('<svg xmlns="http://www.w3.org/2000/svg" width="80" height="40"><rect width="80" height="40" fill="red"/></svg>')
        self.assertEqual(read_image(svg).size, (512, 512))
        svg.write_text('<svg xmlns="http://www.w3.org/2000/svg"><image href="https://untrusted.example/image.png"/></svg>')
        with self.assertRaises(ValueError):
            read_image(svg)


if __name__ == "__main__":
    unittest.main()
