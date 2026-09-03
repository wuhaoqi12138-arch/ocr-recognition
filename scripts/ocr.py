#!/usr/bin/env python3
"""OCR images / image-only PDFs with RapidOCR preinstalled on the runtime.

Never pip/apt install OCR packages from this script. RapidOCR is only on
runtime rt-18 (image selfhost-ocr). If RapidOCR is missing, exit 2 and tell
the operator to bind the agent to rt-18.

Files are processed one by one. Native OCR threads default to 1
(OCR_INTRA_OP_THREADS / ORT_INTRA_OP_NUM_THREADS) so a task cannot saturate
the host; Docker --cpus is the hard cap.

Usage:
  ocr.py FILE [FILE ...] -o OUT_DIR [--json] [--min-score 0.5]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
PDF_EXTS = {".pdf"}


def _fail(msg: str, code: int = 2) -> None:
    print(msg, file=sys.stderr)
    raise SystemExit(code)


def ocr_thread_count() -> int:
    """Intra-op threads for RapidOCR/onnxruntime. Default 1 so one task cannot saturate the host."""
    raw = os.environ.get("OCR_INTRA_OP_THREADS") or os.environ.get(
        "ORT_INTRA_OP_NUM_THREADS"
    ) or "1"
    try:
        n = int(raw)
    except ValueError:
        n = 1
    return max(1, n)


def limit_native_threads(n: int | None = None) -> int:
    """Cap OpenMP/onnxruntime threads. Must run before importing cv2/onnxruntime."""
    n = ocr_thread_count() if n is None else max(1, n)
    os.environ["OMP_NUM_THREADS"] = str(n)
    os.environ["OPENBLAS_NUM_THREADS"] = str(n)
    os.environ["MKL_NUM_THREADS"] = str(n)
    os.environ["NUMEXPR_NUM_THREADS"] = str(n)
    os.environ["ORT_INTRA_OP_NUM_THREADS"] = str(n)
    os.environ["ORT_INTER_OP_NUM_THREADS"] = "1"
    return n


def load_engine():
    """Return RapidOCR with thread caps. Do not pip-install on ImportError."""
    n = limit_native_threads()
    try:
        from rapidocr_onnxruntime import RapidOCR
    except ImportError:
        _fail(
            "RapidOCR is not installed in this runtime. "
            "Do NOT pip install / apt install tesseract or rapidocr. "
            "Rebuild the OCR image (INSTALL_OCR=1 → multica-employee:selfhost-ocr) "
            "and bind this agent to runtime rt-18. Do NOT pip/apt install OCR here."
        )
    try:
        return RapidOCR(intra_op_num_threads=n, inter_op_num_threads=1)
    except TypeError:
        return RapidOCR()


def _sorted_items(raw: Any, min_score: float) -> list[dict]:
    """Normalize RapidOCR rows: [box, text, score] → reading-order dicts."""
    items: list[dict] = []
    if not raw:
        return items
    for row in raw:
        if not row or len(row) < 2:
            continue
        box, text = row[0], str(row[1] or "").strip()
        score = float(row[2]) if len(row) > 2 and row[2] is not None else 0.0
        if not text or score < min_score:
            continue
        ys = [p[1] for p in box] if box else [0]
        xs = [p[0] for p in box] if box else [0]
        items.append(
            {
                "text": text,
                "score": round(score, 4),
                "box": box,
                "y": min(ys),
                "x": min(xs),
            }
        )
    items.sort(key=lambda it: (it["y"], it["x"]))
    return items


def ocr_image(engine, path: str, min_score: float) -> list[dict]:
    raw, _elapsed = engine(path)
    return _sorted_items(raw, min_score)


def ocr_pdf(engine, path: str, min_score: float, dpi: float = 144.0) -> list[dict]:
    try:
        import fitz
        import numpy as np
    except ImportError:
        _fail(
            "PyMuPDF/numpy is not installed in this runtime. "
            "Do NOT pip install it. Rebuild the employee image."
        )
    doc = fitz.open(path)
    pages: list[dict] = []
    zoom = dpi / 72.0
    matrix = fitz.Matrix(zoom, zoom)
    try:
        for i, page in enumerate(doc, start=1):
            pix = page.get_pixmap(matrix=matrix, alpha=False)
            img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
                pix.height, pix.width, pix.n
            )
            raw, _elapsed = engine(img)
            pages.append({"page": i, "items": _sorted_items(raw, min_score)})
    finally:
        doc.close()
    return pages


def _lines(items: list[dict]) -> str:
    return "\n".join("%s  [%.2f]" % (it["text"], it["score"]) for it in items)


def _stem(path: str) -> str:
    return os.path.splitext(os.path.basename(path))[0]


def write_image_result(out_dir: str, src: str, items: list[dict], as_json: bool) -> str:
    base = os.path.join(out_dir, _stem(src))
    txt = base + ".txt"
    with open(txt, "w", encoding="utf-8") as fh:
        fh.write(_lines(items))
        fh.write("\n")
    if as_json:
        payload = {"file": src, "pages": [{"page": 1, "items": items}]}
        with open(base + ".json", "w", encoding="utf-8") as fh:
            json.dump(payload, fh, ensure_ascii=False, indent=2)
    return txt


def write_pdf_result(out_dir: str, src: str, pages: list[dict], as_json: bool) -> str:
    base = os.path.join(out_dir, _stem(src))
    txt = base + ".txt"
    chunks: list[str] = []
    for page in pages:
        chunks.append("===== page %d =====" % page["page"])
        chunks.append(_lines(page["items"]))
    with open(txt, "w", encoding="utf-8") as fh:
        fh.write("\n".join(chunks).rstrip() + "\n")
    if as_json:
        with open(base + ".json", "w", encoding="utf-8") as fh:
            json.dump({"file": src, "pages": pages}, fh, ensure_ascii=False, indent=2)
    return txt


def classify(path: str) -> str:
    ext = os.path.splitext(path)[1].lower()
    if ext in IMAGE_EXTS:
        return "image"
    if ext in PDF_EXTS:
        return "pdf"
    return ""


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="OCR images/PDFs with preinstalled RapidOCR")
    p.add_argument("files", nargs="+", help="Image or PDF paths")
    p.add_argument("-o", "--output-dir", required=True, help="Directory for .txt / .json")
    p.add_argument("--json", action="store_true", help="Also write per-file JSON with boxes")
    p.add_argument("--min-score", type=float, default=0.5, help="Drop items below this score")
    return p.parse_args(argv)


def run(args: argparse.Namespace, engine=None) -> list[dict]:
    """OCR files one after another. Do not parallelize — RapidOCR is CPU-heavy."""
    os.makedirs(args.output_dir, exist_ok=True)
    if engine is None:
        engine = load_engine()
    summary: list[dict] = []
    for path in args.files:
        if not os.path.isfile(path):
            _fail("File not found: " + path)
        kind = classify(path)
        if not kind:
            _fail("Unsupported file type: " + path + " (use jpg/png/webp/bmp/tiff/pdf)")
        if kind == "image":
            items = ocr_image(engine, path, args.min_score)
            out = write_image_result(args.output_dir, path, items, args.json)
            summary.append({"file": path, "pages": 1, "items": len(items), "output": out})
        else:
            pages = ocr_pdf(engine, path, args.min_score)
            n = sum(len(p["items"]) for p in pages)
            out = write_pdf_result(args.output_dir, path, pages, args.json)
            summary.append(
                {"file": path, "pages": len(pages), "items": n, "output": out}
            )
    return summary


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    summary = run(args)
    print("OK %d file(s) -> %s" % (len(summary), args.output_dir))
    for row in summary:
        print(
            "%s  pages=%d items=%d  %s"
            % (row["file"], row["pages"], row["items"], row["output"])
        )


if __name__ == "__main__":
    main()
