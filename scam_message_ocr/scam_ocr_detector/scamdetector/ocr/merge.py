from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, List

@dataclass
class OCRPick:
    text: str
    engine: str
    avg_conf: float
    reason: str


def choose_best_text(
    user_text: Optional[str],
    ocr_text: str,
    ocr_engine: str,
    avg_conf: float,
) -> str:
    """
    원칙:
    - 사용자가 붙여넣은 text가 있으면 그것을 1순위로
    - OCR은 "누락 보완"으로 뒤에 붙임 (중복이 심하면 나중에 dedup 개선 가능)
    """
    user_text = (user_text or "").strip()
    ocr_text = (ocr_text or "").strip()

    if user_text and ocr_text:
        return user_text + "\n\n[OCR_EXTRACTED]\n" + ocr_text
    if user_text:
        return user_text
    return ocr_text