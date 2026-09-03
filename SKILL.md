---
name: ocr-recognition
description: Use when extracting text from photos, scans, screenshots, work tickets, forms, or image-only PDFs; or when the model reports it cannot see images / omits image content from Read. Triggers: OCR, RapidOCR, tesseract, 文字识别, 扫描件, 工作票图片, 模型不支持图片.
---

## ⚠️ 硬性规则（先读）

运行时 **仅 rt-18** 预装 RapidOCR（镜像 `multica-employee:selfhost-ocr`）。rt-01…rt-17 **没有** OCR 依赖。需要读图的智能体必须绑定 **rt-18**。

**禁止：**

- `pip install` rapidocr / pytesseract / opencv / pillow / pymupdf（OCR 用途）
- `apt` / `sudo` 安装 tesseract 或其它 OCR 包
- 建 venv 再装一遍 OCR
- 用 Read 工具指望模型看懂图片（当前模型常会省略图片）
- 自己写 `ocr_parallel.py` / `ProcessPool` / `ThreadPool` / 多 worker 并行 OCR（会打满宿主机 CPU）

缺依赖时：**停止并报告「请把智能体绑到 rt-18」**，不要自行安装。

OCR **必须串行**：一次调用 `ocr.py`，把多张图作为位置参数传入即可（脚本内部逐张处理；线程数由 rt-18 的 `OCR_INTRA_OP_THREADS` 控制）。

---

# OCR Recognition

## When to use

- 附件是 jpg/png/扫描 PDF，需要读上面的字
- 模型提示 `Current model does not support images` / 图片被 omitted
- 工作票、表单、证件、截图等印刷/手写文字提取

**不要用本技能：** 已有文本层的 PDF（先 `pymupdf`/`fitz` 抽文本）；生成新图片（用 image-generation）。

## Workflow

### 1. 下载附件

```bash
multica attachment download <attachment-id>
```

### 2. 调用预装 RapidOCR（不要读脚本源码）

```bash
S=$(find "$PWD" -path "*/.pi/skills/ocr-recognition/scripts/ocr.py" | head -1)
python3 "$S" ./票.jpg ./扫描件.pdf -o ./ocr_out --json
```

| 参数 | 说明 |
|------|------|
| 位置参数 | 一个或多个 jpg/jpeg/png/webp/bmp/tif/tiff/pdf |
| `-o ./ocr_out` | 输出目录（写 `.txt`；加 `--json` 再写带框的 `.json`） |
| `--min-score 0.5` | 丢弃低于该置信度的行 |

stdout 示例：`OK 2 file(s) -> ./ocr_out`。然后 **Read 生成的 `.txt`**，不要再对原图做视觉理解。

多文件请仍用这一条命令（脚本逐张串行）。**禁止** `nproc` 后按核数开 8 个进程并行 OCR。

PDF 无文本层时由脚本渲染再 OCR；有文本层的 PDF 可先试 `python3 -c "import fitz; ... get_text()"`，抽到正文就不必 OCR。

### 3. 交付

把识别结果写进回复 / `multica issue comment add`。需要归档时 `multica attachment upload ./ocr_out/...`。

## Errors

| 情况 | 行为 |
|------|------|
| RapidOCR / fitz 无法 import | 退出 2，**禁止 pip**；当前不是 rt-18，请改绑 OCR 运行时 |
| 文件不存在 / 不支持的后缀 | 退出 2 |
| 成功 | 退出 0，目录内有 `.txt` |

## Common mistakes

| 借口 | 正确做法 |
|------|----------|
| 「环境没有 OCR，先 pip 一下」 | 只有 rt-18 预装；绑错运行时时改绑，不要 pip |
| 「先装 tesseract 更准」 | 无 sudo，不要装；用 RapidOCR |
| 「Read 图片即可」 | 模型常看不到图；必须跑 `ocr.py` |
| 「我自己写一段 RapidOCR」 | 调用本技能脚本，保持输出路径一致 |
| 「933 张太慢，开 8 worker」 | 禁止。并行会把整机打满；用 `ocr.py` 一次传入多个文件 |
