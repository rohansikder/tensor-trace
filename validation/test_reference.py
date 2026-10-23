"""Checks for the independent reference and committed numeric fixtures."""

import hashlib
import json
from pathlib import Path
import struct
import tempfile
import unittest

import numpy as np

from reference import Checkpoint, forward, rms_norm, rotary, softmax

FIXTURES = Path(__file__).resolve().parent / "fixtures"


class ReferenceTests(unittest.TestCase):
    def test_rms_norm_known_vector(self):
        actual = rms_norm(np.array([[3.0, 4.0]], dtype=np.float32), np.array([2.0, 0.5], dtype=np.float32))
        expected = np.array([[6.0, 2.0]], dtype=np.float32) / np.sqrt(np.float32(12.50001))
        np.testing.assert_allclose(actual, expected, atol=1e-7)

    def test_softmax_is_stable_and_masks_future_positions(self):
        scores = np.array([[10000.0, 10001.0, -np.inf]], dtype=np.float32)
        result = softmax(scores)
        np.testing.assert_allclose(result, [[0.26894143, 0.7310586, 0]], atol=1e-7)
        self.assertEqual(float(result[0, 2]), 0.0)

    def test_rotary_position_zero_and_rotation_norm(self):
        values = np.arange(24, dtype=np.float32).reshape(3, 2, 4) / 10
        rotated = rotary(values)
        np.testing.assert_array_equal(rotated[0], values[0])
        np.testing.assert_allclose(np.sum(rotated * rotated, axis=-1), np.sum(values * values, axis=-1), rtol=1e-6)

    def test_grouped_query_fixture_shape_and_classifier_modes(self):
        shared = Checkpoint(FIXTURES / "tiny-shared.bin")
        unshared = Checkpoint(FIXTURES / "tiny-unshared.bin")
        self.assertTrue(shared.config.shared_classifier)
        self.assertFalse(unshared.config.shared_classifier)
        self.assertEqual(shared.config.n_heads, 4)
        self.assertEqual(shared.config.n_kv_heads, 2)
        self.assertEqual(shared.wk.shape, (2, 8, 16))
        self.assertEqual(unshared.classifier.shape, (512, 16))

    def test_attention_is_causal_with_normalized_rows(self):
        model = Checkpoint(FIXTURES / "tiny-shared.bin")
        result = forward(model, [1, 7, 29, 4, 30], layer=1, head=3)
        matrix = np.asarray(result["attention"])
        np.testing.assert_array_equal(matrix[np.triu_indices(5, 1)], 0)
        np.testing.assert_allclose(matrix.sum(axis=1), np.ones(5), atol=1e-7)

    def test_future_tokens_cannot_change_prior_attention(self):
        model = Checkpoint(FIXTURES / "tiny-shared.bin")
        prefix = np.asarray(forward(model, [1, 7, 29], layer=1, head=2)["attention"])
        extended = np.asarray(forward(model, [1, 7, 29, 11, 145], layer=1, head=2)["attention"])
        np.testing.assert_allclose(prefix, extended[:3, :3], atol=1e-6)

    def test_fixtures_match_their_hashes_and_independent_goldens(self):
        goldens = json.loads((FIXTURES / "goldens.json").read_text())
        for item in goldens["models"]:
            path = FIXTURES / item["file"]
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), item["sha256"])
            model = Checkpoint(path)
            for case in item["cases"]:
                with self.subTest(model=path.name, tokens=len(case["tokens"])):
                    actual = forward(model, case["tokens"], case["layer"], case["head"])
                    np.testing.assert_allclose(actual["logits"], case["logits"], atol=2e-5, rtol=2e-5)
                    np.testing.assert_allclose(actual["attention"], case["attention"], atol=2e-5, rtol=2e-5)
                    self.assertEqual(actual["greedyToken"], case["greedyToken"])

    def test_invalid_tokens_and_context_are_rejected(self):
        model = Checkpoint(FIXTURES / "tiny-shared.bin")
        for tokens in ([], [-1], [512], [1] * 33):
            with self.subTest(tokens=tokens), self.assertRaises(ValueError):
                forward(model, tokens)

    def test_invalid_checkpoint_layout_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.bin"
            for content in (b"short", struct.pack("<7i", 16, 32, 2, 4, 2, 512, 32), struct.pack("<7i", 15, 32, 2, 4, 2, 512, 32)):
                path.write_bytes(content)
                with self.subTest(size=len(content)), self.assertRaises((ValueError, OSError)):
                    Checkpoint(path)


if __name__ == "__main__":
    unittest.main()
