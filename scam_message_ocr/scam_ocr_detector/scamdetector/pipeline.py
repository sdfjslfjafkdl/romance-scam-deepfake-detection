from __future__ import annotations
from dataclasses import asdict
from typing import Optional, Dict, Any
from PIL import Image

from scamdetector.config import AppConfig
from scamdetector.ocr.ocr_paddle import PaddleOCREngine
from scamdetector.ocr.merge import choose_best_text
from scamdetector.features.rules import extract_rule_signals

def _make_qwen_analyzer(cfg: AppConfig):
    from scamdetector.llm.qwen_transformers import QwenTransformersAnalyzer
    return QwenTransformersAnalyzer(
        model_name=cfg.qwen.model_name,
        device=cfg.qwen.device,
        max_new_tokens=cfg.qwen.max_new_tokens,
        temperature=cfg.qwen.temperature,
    )

def run_pipeline(
    cfg: AppConfig,
    text: Optional[str] = None,
    image: Optional[Image.Image] = None,
) -> Dict[str, Any]:

    ocr_text = ""
    ocr_engine_name = None
    ocr_avg_conf = 0.0

    if image is not None:
        paddle = PaddleOCREngine(lang=cfg.ocr.lang, use_angle_cls=cfg.ocr.cls)
        ocr_res = paddle.run(image)
        ocr_text = ocr_res.merged_text
        ocr_engine_name = ocr_res.engine
        ocr_avg_conf = ocr_res.avg_conf

        if cfg.ocr.use_easyocr_fallback and (
            len(ocr_text.strip()) < cfg.ocr.min_text_length_ok
            or ocr_avg_conf < cfg.ocr.min_avg_conf_ok
        ):
            try:
                from scamdetector.ocr.ocr_easy import EasyOCREngine
                easy = EasyOCREngine(langs=["ko","en"])
                easy_res = easy.run(image)
                if (len(easy_res.merged_text.strip()) > len(ocr_text.strip())) or (easy_res.avg_conf > ocr_avg_conf):
                    ocr_text = easy_res.merged_text
                    ocr_engine_name = easy_res.engine
                    ocr_avg_conf = easy_res.avg_conf
            except Exception:
                pass

    merged_text = choose_best_text(
        user_text=text,
        ocr_text=ocr_text,
        ocr_engine=ocr_engine_name or "",
        avg_conf=ocr_avg_conf
    )

    signals = extract_rule_signals(merged_text)
    rule_summary = (
        f"rule_risk_score={signals.rule_risk_score}, "
        f"urls={signals.urls[:3]}, phones={signals.phones[:3]}, "
        f"money_mentions={signals.money_mentions[:3]}, otp_mentions={signals.otp_mentions[:3]}, "
        f"offplatform={signals.has_offplatform}, urgency={signals.has_urgency}, "
        f"romance={signals.has_romance_language}, finance_lang={signals.has_finance_request_language}"
    )

    analyzer = _make_qwen_analyzer(cfg)
    llm_out = analyzer.analyze(merged_text=merged_text, rule_summary=rule_summary)

    return {
        "input_meta": {
            "has_user_text": bool((text or "").strip()),
            "has_image": image is not None,
            "ocr_engine_used": ocr_engine_name,
            "ocr_avg_conf": ocr_avg_conf,
            "qwen_mode": cfg.qwen.mode,
            "qwen_model": cfg.qwen.vllm_model if cfg.qwen.mode == "vllm" else cfg.qwen.model_name
        },
        "rule_signals": asdict(signals),
        "llm": llm_out,
    }