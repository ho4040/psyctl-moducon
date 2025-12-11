# Psyctl Steering Chat

Psyctl 프레임워크로 만들어진 스티어링 벡터를 사용하여 LLM을 스티어링하고 대화할 수 있는 Gradio 앱입니다.

## 기능

- 스티어링 벡터 선택 (드롭다운)
- 스티어링 강도 조절 (-5 ~ 5)
- 벡터에 맞는 모델 자동 로드
- 실시간 채팅 인터페이스
- CPU/GPU 자동 감지

## 사용 가능한 스티어링 벡터

| 벡터 | 설명 | 대상 모델 |
|------|------|----------|
| English - Agreeableness | 친화성 | Llama-3.1-8B-Instruct |
| English - Awfully Sweet | 지나치게 다정함 | Llama-3.1-8B-Instruct |
| English - Neuroticism | 신경증 | Llama-3.1-8B-Instruct |
| English - Paranoid | 편집증 | Llama-3.1-8B-Instruct |
| Korean - Awfully Sweet | 지나치게 다정함 (한국어) | EXAONE-3.5-7.8B-Instruct |
| Korean - Rude | 무례함 (한국어) | EXAONE-3.5-7.8B-Instruct |

## 설치 및 실행

### 1. 로컬 환경 (Windows)

```powershell
# uv 설치 (없는 경우)
Invoke-WebRequest -Uri "https://astral.sh/uv/install.ps1" -OutFile "install_uv.ps1"
& .\install_uv.ps1

# 가상환경 생성 및 의존성 설치
uv venv
.\.venv\Scripts\Activate.ps1
uv sync

# HuggingFace 토큰 설정 (Llama 모델 접근용)
$env:HF_TOKEN = "your_huggingface_token"

# 앱 실행
python app.py
```

### 2. Google Colab / RunPod 환경

```python
# 의존성 설치
!pip install torch transformers gradio safetensors huggingface-hub accelerate

# HuggingFace 토큰 설정
import os
os.environ['HF_TOKEN'] = 'your_huggingface_token'

# 앱 실행
!python app.py
```

### 3. RunPod Docker 배포

#### Docker 이미지 빌드 및 푸시

```bash
# Docker 이미지 빌드
docker build -t your-dockerhub-username/psyctl-steering:latest .

# Docker Hub에 푸시
docker push your-dockerhub-username/psyctl-steering:latest
```

#### RunPod에서 실행

1. RunPod에서 새 Pod 생성
2. Docker 이미지: `your-dockerhub-username/psyctl-steering:latest`
3. 환경 변수 설정:
   - `HF_TOKEN`: HuggingFace 토큰 (필수, Llama 모델 접근용)
4. 포트: 7860 노출
5. GPU: 24GB VRAM 이상 권장 (RTX 3090, RTX 4090, A10G 등)

#### 환경 변수

| 변수명 | 필수 | 설명 |
|--------|------|------|
| `HF_TOKEN` | O | HuggingFace 토큰 (Llama 모델 접근용) |

## 사용 방법

1. 앱 실행 후 브라우저에서 `http://localhost:7860` 접속
2. 스티어링 벡터 드롭다운에서 원하는 벡터 선택
3. (로컬 테스트 시) "CPU 강제 사용" 체크
4. 강도 슬라이더로 스티어링 강도 조절 (-5 ~ 5)
5. "스티어링 적용" 버튼 클릭 (해당 모델이 자동으로 로드됨)
6. 채팅창에서 메시지 입력 후 대화

## 프로젝트 구조

```
psyctl-moducon/
├── app.py                  # Gradio 메인 앱
├── pyproject.toml          # 프로젝트 설정 및 의존성
├── Dockerfile              # RunPod용 Docker 이미지
├── start.sh                # 컨테이너 시작 스크립트
├── README.md
├── src/
│   ├── __init__.py
│   ├── model_loader.py     # LLM 로드 (CPU/GPU 자동 감지)
│   ├── steering.py         # 스티어링 벡터 로드 및 적용
│   └── vector_manager.py   # HuggingFace에서 벡터 다운로드
└── vectors/                # 스티어링 벡터 캐시
```

## 참고

- [Psyctl 프로젝트](https://github.com/modulabs-personalab/psyctl)
- [스티어링 벡터](https://huggingface.co/dalekwon/bipo-steering-vectors)
- 대상 모델
  - English 벡터: meta-llama/Llama-3.1-8B-Instruct
  - Korean 벡터: LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct
