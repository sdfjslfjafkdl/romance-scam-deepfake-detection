# Romance Scam & Deepfake Detection

1️⃣ Real-time Deepfake Detection (Webcam / Video)

```bash
cd deepfake_realtime
conda create -n deepfake311 python=3.11 -y
conda activate deepfake311
pip install -r requirements.txt
python demo_webcam.py
```

2️⃣ OCR-based Scam Message Detection

```bash
cd scam_message_ocr
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python cli.py --text "text" --image path/to/chat_screenshot.png
```