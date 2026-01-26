from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional
import numpy as np
from PIL import Image

try:
    import easyocr
except Exception:
    easyocr = None


@dataclass
class OCRLine:
    text: str
    conf: float
    bbox: Optional[List[List[float]]] = None

@dataclass
class OCRResult:
    lines: List[OCRLine]
    merged_text: str
    avg_conf: float
    engine: str


def _pil_to_rgb_np(image: Image.Image) -> np.ndarray:
    return np.array(image.convert("RGB"))


class EasyOCREngine:
    def __init__(self, langs=None):
        if easyocr is None:
            raise ImportError("easyocr import failed. Please install easyocr + torch.")
        if langs is None:
            langs = ["ko", "en"]
        self.reader = easyocr.Reader(langs, gpu=False)

    def run(self, image: Image.Image) -> OCRResult:
        img = _pil_to_rgb_np(image)
        out = self.reader.readtext(img)

        lines: List[OCRLine] = []
        confs: List[float] = []

        for bbox, text, conf in out:
            text = (text or "").strip()
            conf = float(conf)
            if text:
                lines.append(OCRLine(text=text, conf=conf, bbox=bbox))
                confs.append(conf)

        merged_text = "\n".join([l.text for l in lines])
        avg_conf = float(sum(confs) / len(confs)) if confs else 0.0
        return OCRResult(lines=lines, merged_text=merged_text, avg_conf=avg_conf, engine="easyocr")