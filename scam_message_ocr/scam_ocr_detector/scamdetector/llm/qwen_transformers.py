from __future__ import annotations
import json
import re
from json_repair import repair_json
from dataclasses import dataclass
from typing import Dict, Any, Optional

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from scamdetector.llm.qwen_schema import SCAM_JSON_SCHEMA_DESC

import json, re
from json_repair import repair_json

def _extract_json(text: str):
    text = text.strip()

    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE).strip()
    text = re.sub(r"\s*```$", "", text).strip()

    m = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not m:
        raise ValueError("No JSON object found in output.")
    js = m.group(0).strip()

    try:
        return json.loads(js)
    except json.JSONDecodeError:
        pass

    repaired = repair_json(js)          
    return json.loads(repaired)


@dataclass
class QwenTransformersAnalyzer:
    model_name: str
    device: str = "auto"
    max_new_tokens: int = 900
    temperature: float = 0.2

    _tokenizer: Optional[AutoTokenizer] = None
    _model: Optional[AutoModelForCausalLM] = None

    def _load(self):
        if self._model is not None:
            return
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if self.device == "auto" else None,
            trust_remote_code=True
        )
        if self.device in ("cuda", "cpu"):
            self._model.to(self.device)

    def _generate(self, prompt: str) -> str:
        self._load()
        tok = self._tokenizer
        model = self._model

        inputs = tok(prompt, return_tensors="pt")
        if hasattr(model, "device"):
            inputs = {k: v.to(model.device) for k, v in inputs.items()}

        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=self.temperature > 0,
                temperature=self.temperature,
            )
        decoded = tok.decode(out[0], skip_special_tokens=True)

        if decoded.startswith(prompt):
            decoded = decoded[len(prompt):]
        return decoded.strip()

    def analyze(self, merged_text: str, rule_summary: str) -> Dict[str, Any]:
        prompt = f"""
You are a scam-detection analyst.
You must be cautious: never claim certainty. Focus on risk signals.
You MUST quote exact lines from TEXT as evidence.
Return ONLY JSON. No extra commentary.

{SCAM_JSON_SCHEMA_DESC}

RULE_SIGNALS_SUMMARY:
{rule_summary}

TEXT:
{merged_text}

Notes:
- Use crime script stages: rapport_building -> isolation_or_offplatform -> request_money_or_credentials.
- If insufficient text, populate need_more_context.
- Create 3-5 likely next scammer messages (simulations).
""".strip()

        out = self._generate(prompt)
        try:
            return _extract_json(out)
        except Exception:
            repair_prompt = prompt + "\n\nYour previous output was invalid JSON. Output ONLY valid JSON now."
            out2 = self._generate(repair_prompt)
            return _extract_json(out2)