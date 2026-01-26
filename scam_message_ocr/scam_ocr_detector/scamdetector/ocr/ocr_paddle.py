from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional
import numpy as np
from PIL import Image

from paddleocr import PaddleOCR

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

def _pil_to_bgr_np(image: Image.Image) -> np.ndarray:
    rgb = np.array(image.convert("RGB"))
    return rgb[:, :, ::-1].copy()

class PaddleOCREngine:
    def __init__(self, lang: str = "korean", use_angle_cls: bool = True):
        self.ocr = PaddleOCR(use_angle_cls=use_angle_cls, lang=lang)

    def run(self, image: Image.Image) -> OCRResult:
        img = _pil_to_bgr_np(image)
        raw = self.ocr.ocr(img, cls=True)  # 2.7.x에서는 OK

        lines: List[OCRLine] = []
        confs: List[float] = []
        if not raw:
            return OCRResult([], "", 0.0, "paddleocr")

        # raw: [ [ [bbox, (text, conf)], ... ] ] 형태가 흔함
        for item in raw[0]:
            bbox = item[0]
            text, conf = item[1][0], float(item[1][1])
            text = (text or "").strip()
            if text:
                lines.append(OCRLine(text=text, conf=conf, bbox=bbox))
                confs.append(conf)

        merged = "\n".join([l.text for l in lines])
        avg = float(sum(confs) / len(confs)) if confs else 0.0
        return OCRResult(lines, merged, avg, "paddleocr")