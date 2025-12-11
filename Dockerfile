# RunPod용 Dockerfile
# Ubuntu 22.04 + CUDA 12.5 기반 (Google Colab 환경과 동일)

FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

# 작업 디렉토리 설정
WORKDIR /app

# 시스템 패키지 업데이트
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    && rm -rf /var/lib/apt/lists/*

# Python 패키지 설치
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
    torch \
    transformers \
    gradio \
    safetensors \
    huggingface-hub \
    accelerate

# 애플리케이션 코드 복사
COPY . /app

# HuggingFace 캐시 디렉토리 설정
ENV HF_HOME=/app/.cache/huggingface
ENV TRANSFORMERS_CACHE=/app/.cache/huggingface/transformers

# 벡터 캐시 디렉토리 생성
RUN mkdir -p /app/vectors /app/.cache/huggingface

# Gradio 포트 노출
EXPOSE 7860

# 환경 변수 설명
# HF_TOKEN: 런팟 Pod 생성 시 환경 변수로 설정 필요 (Llama 모델 접근용)
# 예: HF_TOKEN=hf_xxxxx

# 시작 스크립트 실행 권한 부여
RUN chmod +x /app/start.sh

# 앱 실행
CMD ["/app/start.sh"]
