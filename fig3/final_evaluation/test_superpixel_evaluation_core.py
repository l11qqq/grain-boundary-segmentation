"""Headless regression tests for the manual evaluator's counting rules."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from superpixel_evaluation_core import (
    CropRect,
    ReferenceData,
    compute_auto_crossing_ids,
    compute_boundaries,
    evaluate_crop,
    rasterize_labelme_grains,
    save_lossless_crop_png,
)


class SuperpixelEvaluationCoreTests(unittest.TestCase):
    def test_auto_mark_and_manual_override(self) -> None:
        candidate = np.array(
            [
                [1, 1, 2, 2],
                [1, 3, 3, 2],
                [4, 4, 3, 3],
                [4, 4, 3, 3],
            ],
            dtype=np.int32,
        )
        reference_labels = np.array(
            [
                [10, 20, 20, 20],
                [10, 10, 10, 20],
                [30, 30, 10, 10],
                [30, 30, 10, 10],
            ],
            dtype=np.int32,
        )
        reference = ReferenceData(
            reference_labels,
            np.ones(reference_labels.shape, dtype=bool),
            compute_boundaries(reference_labels),
            {},
        )
        automatic, diagnostics = compute_auto_crossing_ids(candidate, reference)
        self.assertEqual(automatic, {1})
        self.assertEqual(diagnostics[1]["dominant_reference_pixels"], 2)
        self.assertEqual(diagnostics[1]["other_reference_pixels"], 1)

        crop = CropRect(0, 0, 3, 2)
        result = evaluate_crop(candidate, crop, automatic)
        self.assertEqual(result.crop_superpixel_ids, {1, 2, 3})
        self.assertEqual(result.final_marked_ids, {1})

        corrected = evaluate_crop(
            candidate,
            crop,
            automatic,
            manual_marked_ids={3},
            manual_unmarked_ids={1},
        )
        self.assertEqual(corrected.final_marked_ids, {3})
        self.assertAlmostEqual(corrected.marked_ratio, 1 / 3)

    def test_crop_border_regions_are_counted_once(self) -> None:
        candidate = np.array(
            [
                [1, 1, 1, 2, 2, 2],
                [1, 1, 1, 2, 2, 2],
                [3, 3, 4, 4, 5, 5],
                [3, 3, 4, 4, 5, 5],
                [6, 6, 6, 7, 7, 7],
                [6, 6, 6, 7, 7, 7],
            ],
            dtype=np.int32,
        )
        result = evaluate_crop(candidate, CropRect(1, 1, 5, 5), global_auto_crossing_ids={1, 7})
        self.assertEqual(result.crop_superpixel_ids, {1, 2, 3, 4, 5, 6, 7})
        self.assertEqual(result.total_superpixels, 7)
        self.assertEqual(result.final_marked_ids, {1, 7})
        self.assertTrue({1, 2, 3, 5, 6, 7}.issubset(result.border_touching_superpixel_ids))

    def test_sparse_pair_aggregation_handles_many_region_ids(self) -> None:
        # A dense 40,000×40,000 overlap matrix would require many GB.  Each
        # candidate here matches one reference region, so the sparse result is
        # well-defined and contains no crossing regions.
        size = 200
        candidate = np.arange(size * size, dtype=np.int64).reshape(size, size)
        reference_labels = candidate + 100_000
        reference = ReferenceData(
            reference_labels,
            np.ones(reference_labels.shape, dtype=bool),
            compute_boundaries(reference_labels),
            {},
        )
        automatic, diagnostics = compute_auto_crossing_ids(candidate, reference)
        self.assertEqual(automatic, set())
        self.assertEqual(len(diagnostics), size * size)

    def test_labelme_overlap_is_excluded(self) -> None:
        payload = {
            "imageHeight": 10,
            "imageWidth": 10,
            "imagePath": "dummy.png",
            "shapes": [
                {
                    "label": "grain",
                    "shape_type": "polygon",
                    "points": [[1, 1], [6, 1], [6, 6], [1, 6]],
                },
                {
                    "label": "grain",
                    "shape_type": "polygon",
                    "points": [[4, 4], [8, 4], [8, 8], [4, 8]],
                },
            ],
        }
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "grains.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            reference = rasterize_labelme_grains(path)
        self.assertFalse(reference.valid_mask[5, 5])
        self.assertEqual(reference.labels[5, 5], 0)
        self.assertEqual(reference.metadata["grain_polygon_count"], 2)
        self.assertGreater(reference.metadata["ambiguous_overlap_pixels_excluded"], 0)

    def test_lossless_crop_png_preserves_exact_source_pixels(self) -> None:
        image_rgb = np.arange(6 * 8 * 3, dtype=np.uint8).reshape(6, 8, 3)
        crop = CropRect(2, 1, 7, 5)
        expected = image_rgb[1:5, 2:7].copy()
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "exact_crop.png"
            returned = save_lossless_crop_png(path, image_rgb, crop)
            with Image.open(path) as saved_image:
                self.assertEqual(saved_image.format, "PNG")
                self.assertEqual(saved_image.size, (5, 4))
                restored = np.asarray(saved_image.convert("RGB"))
        np.testing.assert_array_equal(returned, expected)
        np.testing.assert_array_equal(restored, expected)


if __name__ == "__main__":
    unittest.main()
