#!/bin/bash
# RunPod 시작 스크립트

echo "=== Psyctl Steering Chat 시작 ==="

# HF_TOKEN 확인
if [ -z "$HF_TOKEN" ]; then
    echo "[WARNING] HF_TOKEN이 설정되지 않음. Llama 모델 접근에 문제가 있을 수 있습니다."
    echo "[INFO] RunPod 환경 변수에 HF_TOKEN을 설정하세요."
fi

# 디렉토리 이동
cd /app

# 앱 실행
echo "[INFO] Gradio 앱 시작 중..."
python app.py
