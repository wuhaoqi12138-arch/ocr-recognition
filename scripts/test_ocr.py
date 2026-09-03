import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import ocr as ocr_mod  # noqa: E402


class FakeEngine:
    """Stand-in for RapidOCR: path → list of [box, text, score]."""

    def __init__(self, mapping=None, default=None):
        self.mapping = mapping or {}
        self.default = default if default is not None else [
            [[[10, 20], [80, 20], [80, 40], [10, 40]], "已执行", 0.98],
            [[[10, 50], [200, 50], [200, 70], [10, 70]], "热力机械工作票", 0.99],
            [[[10, 90], [40, 90], [40, 110], [10, 110]], "噪声", 0.12],
        ]

    def __call__(self, path):
        raw = self.mapping.get(path, self.default)
        return raw, 0.01


class OcrScriptTests(unittest.TestCase):
    def test_sorted_items_drops_low_score_and_orders_by_y(self):
        raw = [
            [[[0, 80], [10, 80], [10, 90], [0, 90]], "第二行", 0.9],
            [[[0, 10], [10, 10], [10, 20], [0, 20]], "第一行", 0.9],
            [[[0, 40], [10, 40], [10, 50], [0, 50]], "丢弃", 0.2],
        ]
        items = ocr_mod._sorted_items(raw, min_score=0.5)
        self.assertEqual([it["text"] for it in items], ["第一行", "第二行"])

    def test_run_image_writes_txt_and_json(self):
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            img = tmp / "ticket.jpg"
            img.write_bytes(b"fake")
            out = tmp / "ocr_out"
            args = ocr_mod.parse_args(
                [str(img), "-o", str(out), "--json", "--min-score", "0.5"]
            )
            summary = ocr_mod.run(args, engine=FakeEngine())
            self.assertEqual(summary[0]["items"], 2)
            body = Path(summary[0]["output"]).read_text(encoding="utf-8")
            self.assertIn("热力机械工作票", body)
            self.assertNotIn("噪声", body)
            payload = json.loads((out / "ticket.json").read_text(encoding="utf-8"))
            self.assertEqual(payload["pages"][0]["items"][0]["text"], "已执行")

    def test_unsupported_type_exits(self):
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            bad = tmp / "notes.docx"
            bad.write_bytes(b"x")
            args = ocr_mod.parse_args([str(bad), "-o", str(tmp / "out")])
            with self.assertRaises(SystemExit) as ei:
                ocr_mod.run(args, engine=FakeEngine())
            self.assertEqual(ei.exception.code, 2)

    def test_missing_file_exits(self):
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            args = ocr_mod.parse_args([str(tmp / "nope.jpg"), "-o", str(tmp / "out")])
            with self.assertRaises(SystemExit) as ei:
                ocr_mod.run(args, engine=FakeEngine())
            self.assertEqual(ei.exception.code, 2)

    def test_load_engine_does_not_pip_on_import_error(self):
        real_import = __import__

        def fake_import(name, *a, **k):
            if name == "rapidocr_onnxruntime" or name.startswith("rapidocr_onnxruntime."):
                raise ImportError("simulated missing")
            return real_import(name, *a, **k)

        with mock.patch("builtins.__import__", side_effect=fake_import):
            with self.assertRaises(SystemExit) as ei:
                ocr_mod.load_engine()
        self.assertEqual(ei.exception.code, 2)

    def test_thread_count_defaults_to_one(self):
        env = {
            k: v
            for k, v in os.environ.items()
            if k not in ("OCR_INTRA_OP_THREADS", "ORT_INTRA_OP_NUM_THREADS")
        }
        with mock.patch.dict(os.environ, env, clear=True):
            self.assertEqual(ocr_mod.ocr_thread_count(), 1)

    def test_limit_native_threads_sets_env(self):
        n = ocr_mod.limit_native_threads(1)
        self.assertEqual(n, 1)
        self.assertEqual(os.environ["OMP_NUM_THREADS"], "1")
        self.assertEqual(os.environ["ORT_INTRA_OP_NUM_THREADS"], "1")
        self.assertEqual(os.environ["ORT_INTER_OP_NUM_THREADS"], "1")


if __name__ == "__main__":
    unittest.main()
