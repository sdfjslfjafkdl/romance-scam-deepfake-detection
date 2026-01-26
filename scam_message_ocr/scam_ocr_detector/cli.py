import argparse
import json
from pathlib import Path
from PIL import Image

from scamdetector.config import AppConfig
from scamdetector.pipeline import run_pipeline

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--text", type=str, default=None, help="Copy-pasted message text")
    p.add_argument("--text_file", type=str, default=None, help="Path to a .txt file with message text")
    p.add_argument("--image", type=str, default=None, help="Path to a screenshot image (png/jpg)")
    p.add_argument("--no_easyocr_fallback", action="store_true", help="Disable EasyOCR fallback")
    p.add_argument("--out", type=str, default=None, help="Output JSON path")

    args = p.parse_args()

    text = args.text
    if args.text_file:
        text = Path(args.text_file).read_text(encoding="utf-8")

    image = None
    if args.image:
        image = Image.open(args.image)

    cfg = AppConfig()
    if args.no_easyocr_fallback:
        cfg.ocr.use_easyocr_fallback = False

    result = run_pipeline(cfg=cfg, text=text, image=image)

    out_json = json.dumps(result, ensure_ascii=False, indent=2)
    if args.out:
        Path(args.out).write_text(out_json, encoding="utf-8")
        print(f"Wrote: {args.out}")
    else:
        print(out_json)

if __name__ == "__main__":
    main()