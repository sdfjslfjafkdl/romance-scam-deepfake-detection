# 🟢 GREEN-LIGHT
실시간 멀티모달 딥페이크 기반 로맨스 스캠 탐지 시스템

---

## 📌 프로젝트 개요

GREEN-LIGHT는 영상 통화 환경에서 발생하는  
딥페이크 기반 로맨스 스캠을 **실시간으로 탐지**하는 멀티모달 시스템입니다.

- 얼굴 기반 딥페이크 탐지
- 오디오-비디오 립싱크 이상 탐지 (SyncNet)
- Blink 기반 Liveness 신호 분석
- Green / Yellow / Red 위험 신호 UI 제공

---

## 🧠 핵심 구조

입력:
- 웹캠 영상
- 마이크 오디오

처리:
1. Face Deepfake Detection
2. Audio-Visual Sync Detection (SyncNet)
3. Blink / Liveness 분석

출력:
- 위험 등급 (SAFE / WARNING)
- Deepfake 확률
- AV Sync Offset(ms)
- Risk Report Overlay

---

## 📂 프로젝트 구조 (예시)
```
Romance-Scam-and-Deepfake-Detection/
├── romance-detection/
│   ├── dataset.py
│   └── detector.py
│
├── greenlight_ui.py
├── preprocess.py
├── webcam.py
│
├── syncnet_python/
│   ├── SyncNetInstance.py
│   ├── SyncNetModel.py
│   ├── syncnet_compare_dataset.py
│   ├── syncnet_finetune.py
│   └── webcam_sync_compare.py
│
└── README.md
```

---

## ⚙️ 설치 방법

```bash
git clone https://github.com/sdfjslfjafkdl/Romance-Scam-and-Deepfake-Detection.git
cd Romance-Scam-and-Deepfake-Detection
```

가상환경 생성:
```bash
conda create -n greenlight python=3.11
conda activate greenlight
```

패키지 설치:
```bash
pip install -r requirements.txt
```

🎥 웹캠 실시간 실행
```bash
python romance-detection/main_webcam.py
```
- 실시간 얼굴 분석

- SyncNet 기반 립싱크 검사

- Green / Yellow / Red 신호 표시

- Risk Report 출력

종료: q
