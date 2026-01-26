from dataclasses import dataclass, field

@dataclass
class OCRConfig:
    use_easyocr_fallback: bool = False
    lang: str = "korean"
    cls: bool = True
    min_text_length_ok: int = 15
    min_avg_conf_ok: float = 0.45

@dataclass
class QwenConfig:
    mode: str = "transformers"

    model_name: str = "Qwen/Qwen2.5-7B-Instruct"   
    device: str = "auto"                            
    max_new_tokens: int = 450
    temperature: float = 0.2

@dataclass
class AppConfig:
    ocr: OCRConfig = field(default_factory=OCRConfig)
    qwen: QwenConfig = field(default_factory=QwenConfig)