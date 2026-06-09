# models/

Local cache for model weights. The detector downloads bare model names here on first run
(`yolo11n-seg.pt`, etc.), so they don't clutter the repo root or get committed.

- `*.pt` files are git-ignored.
- Put `yolopv2.pt` here and run with `--yolopv2-model models/yolopv2.pt`.
- SAM3 weights come from the gated HuggingFace repo `facebook/sam3` (needs a token).

Override the location with `python detector/yolo_server.py --models-dir <path>`.
