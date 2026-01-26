from __future__ import annotations
import re
from dataclasses import dataclass, asdict
from typing import List, Dict, Any

URL_RE = re.compile(r"(https?://\S+|www\.\S+)", re.IGNORECASE)
PHONE_RE = re.compile(r"(\+?\d[\d \-]{7,}\d)")
MONEY_RE = re.compile(r"(\d{1,3}(?:,\d{3})+|\d+)\s*(원|₩|만원|달러|\$|€|유로)")
OTP_RE = re.compile(r"(인증번호|OTP|보안코드|verification code|code)\s*[:\-]?\s*\d{3,8}", re.IGNORECASE)

OFFPLATFORM_KEYWORDS = [
    "텔레그램", "telegram", "카톡", "kakao", "왓츠앱", "whatsapp", "라인", "line",
    "dm로", "개인톡", "다른 앱", "다른 메신저", "옮기자"
]
URGENT_KEYWORDS = [
    "지금", "오늘", "당장", "긴급", "마감", "마지막", "서둘러", "빨리",
    "immediately", "urgent", "right now", "last chance"
]
ROMANCE_KEYWORDS = [
    "사랑", "사귀", "보고싶", "연인", "결혼", "운명", "자기야", "baby", "honey", "love"
]
FINANCE_REQUEST_KEYWORDS = [
    "송금", "입금", "돈", "수수료", "계좌", "충전", "투자", "코인", "대출", "선물",
    "wire", "transfer", "fee", "payment", "crypto", "deposit"
]

def _contains_any(text: str, keywords: List[str]) -> bool:
    t = text.lower()
    return any(k.lower() in t for k in keywords)

@dataclass
class RuleSignals:
    urls: List[str]
    phones: List[str]
    money_mentions: List[str]
    otp_mentions: List[str]
    has_offplatform: bool
    has_urgency: bool
    has_romance_language: bool
    has_finance_request_language: bool
    rule_risk_score: int  # 0~100

def extract_rule_signals(text: str) -> RuleSignals:
    text = text or ""
    urls = URL_RE.findall(text)
    phones = PHONE_RE.findall(text)
    money_mentions = MONEY_RE.findall(text)
    otp_mentions = OTP_RE.findall(text)

    has_offplatform = _contains_any(text, OFFPLATFORM_KEYWORDS)
    has_urgency = _contains_any(text, URGENT_KEYWORDS)
    has_romance = _contains_any(text, ROMANCE_KEYWORDS)
    has_finance = _contains_any(text, FINANCE_REQUEST_KEYWORDS)

    score = 0
    if urls: score += 20
    if otp_mentions: score += 30
    if money_mentions: score += 15
    if has_finance: score += 20
    if has_offplatform: score += 15
    if has_urgency: score += 10
    if has_romance: score += 10

    score = min(score, 100)
    return RuleSignals(
        urls=urls,
        phones=phones,
        money_mentions=["".join(m) if isinstance(m, tuple) else str(m) for m in money_mentions],
        otp_mentions=[m if isinstance(m, str) else str(m) for m in otp_mentions],
        has_offplatform=has_offplatform,
        has_urgency=has_urgency,
        has_romance_language=has_romance,
        has_finance_request_language=has_finance,
        rule_risk_score=score
    )