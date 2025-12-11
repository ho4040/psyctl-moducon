"""
스티어링 벡터를 적용한 LLM과 대화하는 Gradio 앱
"""

import os
import sys
from pathlib import Path

print("[DEBUG] === 환경변수 확인 ===", flush=True)
print(f"[DEBUG] HF_HOME={os.environ.get('HF_HOME', '(not set)')}", flush=True)
print(f"[DEBUG] TRANSFORMERS_CACHE={os.environ.get('TRANSFORMERS_CACHE', '(not set)')}", flush=True)
print(f"[DEBUG] VECTOR_CACHE_DIR={os.environ.get('VECTOR_CACHE_DIR', '(not set)')}", flush=True)
print(f"[DEBUG] GRADIO_PORT={os.environ.get('GRADIO_PORT', '(not set, default 7860)')}", flush=True)
print(f"[DEBUG] HF_TOKEN={'(set)' if os.environ.get('HF_TOKEN') else '(not set)'}", flush=True)
print("[DEBUG] ======================", flush=True)

# 포트 설정
GRADIO_PORT = int(os.environ.get("GRADIO_PORT", "7860"))

print("[DEBUG] import gradio 시작...", flush=True)
import gradio as gr
print("[DEBUG] import gradio 완료", flush=True)

from gradio import ChatMessage
import torch

# src 모듈 임포트를 위한 경로 추가
sys.path.insert(0, str(Path(__file__).parent))

print("[DEBUG] import src 모듈 시작...", flush=True)
from src.model_loader import ModelLoader, get_device_info
from src.steering import SteeringManager, SteeringVector
from src.vector_manager import AVAILABLE_VECTORS, VectorManager
print("[DEBUG] import src 모듈 완료", flush=True)

# 벡터 캐시 디렉토리 (환경변수 또는 기본값)
VECTOR_CACHE_DIR = os.environ.get("VECTOR_CACHE_DIR", "./vectors")

# 전역 상태
model_loader: ModelLoader | None = None
steering_manager: SteeringManager | None = None
vector_manager: VectorManager | None = None
current_vector: SteeringVector | None = None
current_model_name: str | None = None  # 현재 로드된 모델 이름
force_cpu_mode: bool = False  # CPU 강제 사용 모드
current_orthogonal: bool = False  # Orthogonal 모드


def initialize_app():
    """앱 초기화: 매니저 로드"""
    global steering_manager, vector_manager

    print("[DEBUG] VectorManager 생성 중...", flush=True)
    vector_manager = VectorManager(cache_dir=VECTOR_CACHE_DIR)
    print("[DEBUG] VectorManager 생성 완료", flush=True)

    print("[DEBUG] SteeringManager 생성 중...", flush=True)
    steering_manager = SteeringManager()
    print("[DEBUG] SteeringManager 생성 완료", flush=True)

    # 디바이스 정보 출력
    print("[DEBUG] get_device_info() 호출 중...", flush=True)
    device_info = get_device_info()
    print(f"[INFO] 디바이스 정보: {device_info}", flush=True)

    return "앱 초기화 완료. 스티어링 벡터를 선택하면 해당 모델이 자동으로 로드됩니다."


def unload_model_and_steering():
    """모델과 스티어링 벡터를 메모리에서 완전히 제거"""
    global model_loader, steering_manager, current_vector, current_model_name

    # 스티어링 먼저 해제 (훅 제거 및 CUDA 캐시 정리)
    if steering_manager is not None:
        steering_manager.unload_all()

    # 스티어링 벡터 참조 해제
    current_vector = None

    # 모델 정리
    if model_loader is not None:
        if model_loader.model is not None:
            del model_loader.model
            model_loader.model = None
        if model_loader.tokenizer is not None:
            del model_loader.tokenizer
            model_loader.tokenizer = None
        model_loader = None

    current_model_name = None

    # CUDA 캐시 정리 (다시 한번)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Python GC 강제 실행
    import gc
    gc.collect()

    # CUDA 캐시 정리 (GC 후 다시)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print("[INFO] 모델 및 스티어링 벡터 메모리 정리 완료")


def load_model_if_needed_generator(model_name: str, force_cpu: bool):
    """필요한 경우 모델 로드 (generator로 진행 상황 반환)"""
    global model_loader, current_model_name, force_cpu_mode

    # 이미 같은 모델이 로드되어 있으면 스킵
    if model_loader is not None and current_model_name == model_name and model_loader.model is not None:
        yield True, f"[INFO] 모델 이미 로드됨: {model_name}"
        return

    # 기존 모델 및 스티어링 완전히 정리
    if model_loader is not None and model_loader.model is not None:
        yield False, f"[PROGRESS] 기존 모델 및 스티어링 정리 중: {current_model_name}..."
        unload_model_and_steering()

    try:
        yield False, f"[PROGRESS] 모델 다운로드/로드 준비 중: {model_name}..."

        model_loader = ModelLoader(model_name=model_name)

        yield False, f"[PROGRESS] 토크나이저 로드 중: {model_name}..."

        # 토크나이저 먼저 로드
        from transformers import AutoTokenizer
        model_loader.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
        )
        if model_loader.tokenizer.pad_token is None:
            model_loader.tokenizer.pad_token = model_loader.tokenizer.eos_token

        yield False, f"[PROGRESS] 모델 가중치 다운로드/로드 중: {model_name}\n(처음 실행 시 다운로드에 시간이 걸립니다)..."

        # 모델 로드
        from transformers import AutoModelForCausalLM
        device = "cpu" if force_cpu else model_loader.device

        load_kwargs = {
            "trust_remote_code": True,
            "low_cpu_mem_usage": True,
        }

        if device == "cuda":
            load_kwargs["torch_dtype"] = torch.bfloat16
            load_kwargs["device_map"] = "auto"
        else:
            load_kwargs["torch_dtype"] = torch.float32

        model_loader.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            **load_kwargs,
        )

        if device == "cpu":
            yield False, f"[PROGRESS] 모델을 CPU로 이동 중..."
            model_loader.model = model_loader.model.to(device)

        model_loader.model.eval()

        current_model_name = model_name
        force_cpu_mode = force_cpu
        device_info = get_device_info()
        yield True, f"[SUCCESS] 모델 로드 완료: {model_name}\n디바이스: {device_info['device']}"

    except Exception as e:
        import traceback
        traceback.print_exc()
        yield False, f"[ERROR] 모델 로드 실패: {e}"


def change_steering_vector_generator(vector_name: str, strength: float, force_cpu: bool, orthogonal: bool):
    """스티어링 벡터 변경 핸들러 (generator로 진행 상황 반환)"""
    global current_vector, steering_manager, vector_manager, model_loader, current_orthogonal

    if steering_manager is None or vector_manager is None:
        yield "[ERROR] 앱이 초기화되지 않음"
        return

    # "None" 선택 시 스티어링 해제
    if vector_name == "None (스티어링 없음)":
        if steering_manager is not None:
            steering_manager.remove_steering()
        current_vector = None
        yield "[INFO] 스티어링 해제됨"
        return

    try:
        yield f"[PROGRESS] 벡터 정보 확인 중: {vector_name}..."

        # 벡터 정보 가져오기
        vector_info = vector_manager.get_vector_info(vector_name)
        if vector_info is None:
            yield f"[ERROR] 벡터 정보를 찾을 수 없음: {vector_name}"
            return

        required_model = vector_info["model"]

        # 필요한 모델 로드 (generator 사용)
        for success, msg in load_model_if_needed_generator(required_model, force_cpu):
            yield msg
            if not success and "[ERROR]" in msg:
                return

        yield f"[PROGRESS] 스티어링 벡터 다운로드 중: {vector_name}..."

        # 벡터 파일 다운로드/캐시에서 가져오기
        vector_path = vector_manager.get_vector_path(vector_name)
        if vector_path is None:
            yield f"[ERROR] 벡터 파일을 찾을 수 없음: {vector_name}"
            return

        yield f"[PROGRESS] 스티어링 벡터 로드 중..."

        # 스티어링 벡터 로드
        current_vector = SteeringVector(vector_path)

        yield f"[PROGRESS] 모델에 스티어링 적용 중..."

        # 모델에 적용
        current_orthogonal = orthogonal
        steering_manager.apply_steering(
            model=model_loader.model,
            steering_vector=current_vector,
            strength=strength,
            orthogonal=orthogonal,
        )

        # 디바이스 정보 가져오기
        device_info = get_device_info()
        device_str = device_info["device"].upper()
        if device_info.get("gpu_name"):
            device_str = f"{device_str} ({device_info['gpu_name']})"

        orth_str = "ON" if orthogonal else "OFF"
        yield f"[SUCCESS] 스티어링 벡터 적용 완료\n벡터: {vector_name}\n모델: {required_model}\n디바이스: {device_str}\n강도: {strength}\nOrthogonal: {orth_str}\n레이어: {current_vector.layer_names}"

    except Exception as e:
        import traceback
        traceback.print_exc()
        yield f"[ERROR] 스티어링 벡터 적용 실패: {e}"


def update_steering_params(strength: float, orthogonal: bool) -> str:
    """스티어링 강도/orthogonal 업데이트 핸들러"""
    global steering_manager, model_loader, current_vector, current_orthogonal

    if model_loader is None or model_loader.model is None:
        return "[ERROR] 모델이 로드되지 않음"

    if steering_manager is None:
        return "[ERROR] 앱이 초기화되지 않음"

    if current_vector is None:
        return "[INFO] 스티어링 벡터가 선택되지 않음"

    try:
        current_orthogonal = orthogonal
        steering_manager.apply_steering(
            model=model_loader.model,
            steering_vector=current_vector,
            strength=strength,
            orthogonal=orthogonal,
        )
        orth_str = "ON" if orthogonal else "OFF"
        return f"[SUCCESS] 스티어링 업데이트\n강도: {strength}\nOrthogonal: {orth_str}"
    except Exception as e:
        return f"[ERROR] 업데이트 실패: {e}"


def extract_text_content(content):
    """Gradio content에서 텍스트만 추출"""
    if isinstance(content, str):
        return content
    elif isinstance(content, list):
        # 멀티모달 content: [{"type": "text", "text": "..."}, ...]
        texts = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                texts.append(item.get("text", ""))
            elif isinstance(item, str):
                texts.append(item)
        return " ".join(texts)
    return str(content) if content else ""


def chat_stream(message: str, history: list, temperature: float, max_tokens: int):
    """채팅 응답 스트리밍 생성 (generator) - messages 형식 사용"""
    global model_loader

    print(f"[DEBUG] chat_stream 시작: message='{message}', history_len={len(history)}")

    if model_loader is None or model_loader.model is None:
        print("[DEBUG] 모델이 로드되지 않음")
        yield "[ERROR] 모델이 로드되지 않음. 스티어링 벡터를 선택하세요."
        return

    try:
        # 채팅 히스토리를 LLM 메시지 형식으로 변환
        messages = []
        for idx, msg in enumerate(history):
            # Gradio ChatMessage 객체 또는 딕셔너리 처리
            if hasattr(msg, "role"):
                role = msg.role
                content = extract_text_content(msg.content)
            else:
                role = msg.get("role", "user")
                content = extract_text_content(msg.get("content", ""))

            print(f"[DEBUG] history[{idx}]: role={role}, content_type={type(content)}, content='{content[:50] if content else ''}...'")

            if content:  # 빈 content는 스킵
                messages.append({"role": role, "content": content})

        messages.append({"role": "user", "content": message})
        print(f"[DEBUG] 최종 messages 수: {len(messages)}")

        # 채팅 템플릿 적용
        prompt = model_loader.apply_chat_template(messages)
        print(f"[DEBUG] 프롬프트 생성 완료, 길이: {len(prompt)}")

        # 스트리밍 응답 생성
        print(f"[DEBUG] generate_stream 시작...")
        token_count = 0
        last_response = ""
        try:
            for partial_response in model_loader.generate_stream(
                prompt=prompt,
                max_new_tokens=max_tokens,
                temperature=temperature,
                do_sample=temperature > 0,
            ):
                token_count += 1
                last_response = partial_response
                if token_count <= 3 or token_count % 10 == 0:
                    print(f"[DEBUG] 토큰 {token_count}: '{partial_response[-20:] if len(partial_response) > 20 else partial_response}'")
                yield partial_response
        except Exception as gen_error:
            print(f"[DEBUG] generate_stream 중 예외: {gen_error}")
            import traceback
            traceback.print_exc()
            raise

        print(f"[DEBUG] generate_stream 종료, 총 토큰: {token_count}, 최종 응답 길이: {len(last_response)}")

    except Exception as e:
        import traceback
        print(f"[DEBUG] 예외 발생: {e}")
        traceback.print_exc()
        yield f"[ERROR] 응답 생성 실패: {e}"


def get_model_info_for_vector(vector_name: str) -> str:
    """선택한 벡터에 대한 모델 정보 반환"""
    if vector_name == "None (스티어링 없음)":
        return "스티어링 없음"

    vector_info = AVAILABLE_VECTORS.get(vector_name)
    if vector_info is None:
        return "알 수 없는 벡터"

    return f"모델: {vector_info['model']}\n레이어: {vector_info['layer']}"


def create_ui():
    """Gradio UI 생성"""
    # 벡터 선택 옵션 (None 포함)
    vector_choices = ["None (스티어링 없음)"] + list(AVAILABLE_VECTORS.keys())

    # CSS 스타일 정의
    custom_css = """
    .section-title {
        padding: 3px;
    }
    """

    with gr.Blocks(title="Psyctl Steering Chat", css=custom_css) as demo:
        with gr.Row():
            with gr.Column(scale=1):
                # 스티어링 설정 섹션
                with gr.Group():
                    gr.Markdown("### 스티어링 벡터", elem_classes=["section-title"])
                    vector_dropdown = gr.Dropdown(
                        choices=vector_choices,
                        value="None (스티어링 없음)",
                        label="스티어링 벡터 선택",
                        info="선택한 벡터에 맞는 모델이 자동으로 로드됩니다",
                    )
                    vector_info_text = gr.Textbox(
                        label="벡터 정보",
                        value="스티어링 없음",
                        interactive=False,
                        lines=2,
                    )
                    force_cpu_checkbox = gr.Checkbox(
                        label="CPU 강제 사용",
                        value=False,
                        info="GPU가 있어도 CPU로 실행 (로컬 테스트용)",
                    )
                    apply_steering_btn = gr.Button("스티어링 적용", variant="primary")
                    steering_status = gr.Textbox(
                        label="상태",
                        value="스티어링 벡터를 선택하고 '스티어링 적용' 버튼을 클릭하세요",
                        interactive=False,
                        lines=5,
                    )
                

            with gr.Column(scale=2):
                chatbot = gr.Chatbot(
                    label="대화",
                    height=500,
                )
                msg_input = gr.Textbox(
                    label="메시지 입력",
                    placeholder="메시지를 입력하세요...",
                    lines=2,
                )
                with gr.Row():
                    send_btn = gr.Button("전송", variant="primary")
                    clear_btn = gr.Button("대화 초기화")
                
                

            with gr.Column(scale=1):
                # 생성 설정
                with gr.Group():
                    gr.Markdown("### 생성 설정", elem_classes=["section-title"])
                    strength_slider = gr.Slider(
                        minimum=-5.0,
                        maximum=5.0,
                        value=1.0,
                        step=0.1,
                        label="스티어링 강도",
                        info="음수: 반대 방향, 양수: 해당 방향으로 스티어링",
                    )
                    orthogonal_checkbox = gr.Checkbox(
                        label="Orthogonal 모드",
                        value=False,
                        info="기존 활성화에서 스티어링 방향 성분 제거 후 추가",
                    )
                    temperature_slider = gr.Slider(
                        minimum=0.0,
                        maximum=2.0,
                        value=0.7,
                        step=0.1,
                        label="Temperature",
                    )
                    max_tokens_slider = gr.Slider(
                        minimum=32,
                        maximum=1024,
                        value=256,
                        step=32,
                        label="Max Tokens",
                    )
                gr.HTML("""
                        <div style="display: flex; flex-direction: row; gap: 12px; padding: 5px; justify-content: center; align-items:baseline;">
                            <div style="display: flex; justify-content: center;">
                                <svg style="width: 100%; max-width: 70px;" viewBox="0 0 81 32" fill="none" xmlns="http://www.w3.org/2000/svg"><path d="M0 29.5789H4.79686V27.2304H0.865776V20.2426H11.1232V27.2304H7.19211V29.5789H11.989V31.6721H0V29.5789ZM8.7237 25.1373V22.3336H3.26527V25.1373H8.72583H8.7237Z" fill="#F7585C"></path><path d="M25.0841 28.9216H20.2872V31.8529H17.8899V28.9216H13.093V26.8284H25.082V28.9216H25.0841ZM24.2906 22.0422H16.1732V23.7354H24.3991V25.7754H13.7758V20.0064H24.2906V22.0464V22.0422Z" fill="#F7585C"></path><path d="M35.1947 30.1958L26.2689 30.7042L26.0307 28.5387L35.0331 28.0494L35.199 30.1979H35.1969L35.1947 30.1958ZM32.491 18.628C33.0739 19.0088 33.5312 19.547 33.8546 20.2383C34.1822 20.9318 34.3417 21.738 34.3417 22.6591C34.3417 23.5802 34.1779 24.3523 33.8546 25.0437C33.527 25.7371 33.0739 26.2732 32.491 26.654C31.9082 27.0347 31.2402 27.2283 30.4915 27.2283C29.7427 27.2411 29.0833 27.0539 28.5174 26.6731C27.9516 26.2923 27.5049 25.7542 27.1815 25.0628C26.8539 24.3693 26.6944 23.5716 26.6944 22.6612C26.6944 21.7507 26.8582 20.9318 27.1815 20.2404C27.5091 19.547 27.9516 19.0109 28.5174 18.6301C29.0833 18.2494 29.7427 18.0558 30.4915 18.0558C31.2402 18.0558 31.9082 18.2472 32.491 18.6301V18.628ZM31.8912 22.6591C31.8912 22.0294 31.7933 21.4934 31.5955 21.0594C31.4849 20.7786 31.3275 20.5744 31.1254 20.4404C30.9233 20.3085 30.7106 20.2404 30.4893 20.2404C30.2681 20.2404 30.0596 20.3085 29.8618 20.4404C29.664 20.5744 29.5044 20.7807 29.3811 21.0594C29.196 21.4955 29.1045 22.0294 29.1045 22.6591C29.1045 23.2887 29.1981 23.8014 29.3811 24.2226C29.5023 24.4906 29.664 24.6927 29.8618 24.8331C30.0596 24.9735 30.266 25.0437 30.4893 25.0437C30.7127 25.0437 30.9233 24.9735 31.1254 24.8331C31.3275 24.6927 31.487 24.4906 31.5955 24.2226C31.7933 23.7737 31.8912 23.2526 31.8912 22.6591ZM38.2005 17.7644V31.8529H35.6372L35.6925 17.7644H38.2005Z" fill="#F7585C"></path><path d="M46.8774 18.3004C47.0667 19.0109 47.1624 19.7342 47.1624 20.4744C47.1624 21.2764 47.0688 22.0124 46.8859 22.6867C46.7008 23.361 46.4328 23.9354 46.0733 24.4076C45.7287 24.8799 45.3245 25.2351 44.8565 25.4712C44.3885 25.7074 43.8844 25.8265 43.3441 25.8265C42.8038 25.8265 42.2954 25.7074 41.821 25.4712C41.3466 25.2351 40.9446 24.8799 40.6127 24.4076C40.2554 23.946 39.9852 23.3759 39.8001 22.6952C39.6151 22.0145 39.5236 21.2764 39.5236 20.4744C39.5236 19.6725 39.6172 18.9364 39.8001 18.2621C39.9852 17.5878 40.2532 17.0134 40.6127 16.5412C40.9446 16.0562 41.3445 15.6988 41.8104 15.467C42.2784 15.2351 42.7889 15.1138 43.342 15.1032C43.8844 15.0904 44.3864 15.2074 44.8544 15.4478C45.3224 15.6924 45.7287 16.0541 46.0712 16.5391C46.4158 17.0007 46.6838 17.5835 46.8731 18.2962H46.8774V18.3004ZM51.9423 29.5959V31.8529H41.67V26.939H44.2141V29.5959H51.9423ZM44.8416 20.4787C44.8416 19.581 44.6991 18.8514 44.4162 18.294C44.2822 18.026 44.1205 17.826 43.9354 17.692C43.7504 17.5601 43.5547 17.4921 43.3441 17.4921C42.8995 17.4921 42.5443 17.7601 42.2762 18.294C42.0061 18.8641 41.8721 19.5938 41.8721 20.4787C41.8721 21.3636 42.0061 22.0868 42.2762 22.6442C42.4102 22.9122 42.5719 23.1143 42.757 23.2547C42.942 23.3951 43.1377 23.4653 43.3483 23.4653C43.5589 23.4653 43.7525 23.3951 43.9397 23.2547C44.1248 23.1143 44.2843 22.9122 44.4205 22.6442C44.7034 22.0868 44.8459 21.3657 44.8459 20.4787H44.8416ZM47.9218 23.1526V20.8956H49.2684V19.8937H47.9218V17.6367H49.2684V14.7607H51.7764V27.1922H49.2854L49.2662 23.1505H47.9197L47.9218 23.1526Z" fill="#F7585C"></path><path d="M64.7056 24.48H65.9585V26.7731H60.9787V31.8529H58.3579V26.7731H53.1569V24.48H62.0827V14.5416H54.0589V12.2293H64.7034V24.4778H64.7056V24.48Z" fill="#F7585C"></path><path d="M66.2435 31.6721V28.9429H72.238V21.6976H75.0055V28.9429H81V31.6721H66.2435ZM69.5726 17.4729C70.2916 16.5391 70.8723 15.5244 71.3148 14.4331C71.67 13.5482 71.9104 12.3889 72.0338 10.9573C72.1551 9.52566 72.2189 8.11744 72.2189 6.73476V0H74.9864V6.73476C74.9864 7.93663 75.0715 9.28954 75.2459 10.7956C75.4161 12.2995 75.699 13.5142 76.0947 14.4353C76.5754 15.5627 77.1476 16.588 77.8198 17.5112C78.4899 18.4344 79.2812 19.1066 80.1917 19.5321L79.7109 22.4442C78.7281 22.24 77.773 21.638 76.8519 20.6403C75.9287 19.6469 75.1842 18.5046 74.6099 17.2177C74.0398 15.9307 73.7505 14.816 73.7505 13.8694V13.6525H73.5654V13.8694C73.5654 14.816 73.2697 15.9286 72.6805 17.2113C72.0891 18.4919 71.3254 19.632 70.3916 20.634C69.4577 21.6359 68.5111 22.24 67.5518 22.4442L67.071 19.5321C68.0176 19.096 68.8515 18.4089 69.5684 17.4751L69.5726 17.4729Z" fill="#F7585C"></path></svg>
                            </div>
                            <div style="display: flex; justify-content: center; padding: 0px;">
                                <svg style="width: 100%; max-width: 90px;" viewBox="0 0 139 22" fill="none" xmlns="http://www.w3.org/2000/svg"><path d="M21 7.33333C27 17.1111 24 22 12 22C0 22 -3 17.1111 3 7.33333C9 -2.44444 15 -2.44444 21 7.33333Z" fill="black"/><g clip-path="url(#clip0_13_533)"><path d="M41.3561 17.653C40.3255 17.912 39.2648 18.0322 38.2024 18.0104C35.0486 18.0104 31.979 16.3284 31.979 11.3245V8.67535C31.979 3.66092 35.0066 1.99996 38.2024 1.99996C39.2648 1.97812 40.3255 2.09832 41.3561 2.35738C41.6504 2.44148 41.8186 2.52558 41.8186 2.81993V4.85934C41.8272 4.91699 41.8223 4.97584 41.8043 5.03129C41.7864 5.08673 41.7559 5.13729 41.7152 5.179C41.6745 5.22072 41.6247 5.25247 41.5697 5.27179C41.5147 5.2911 41.456 5.29745 41.3982 5.29035C40.5992 5.22727 39.4428 5.12215 38.2444 5.12215C36.7937 5.12215 35.5953 5.87905 35.5953 8.67535V11.3245C35.5953 14.1208 36.7937 14.8672 38.2444 14.8672C39.4008 14.8672 40.5572 14.7621 41.3982 14.699C41.455 14.6922 41.5126 14.6984 41.5667 14.7172C41.6208 14.7359 41.67 14.7667 41.7105 14.8072C41.7509 14.8477 41.7817 14.8968 41.8004 14.9509C41.8192 15.005 41.8254 15.0626 41.8186 15.1195V17.1589C41.8186 17.4638 41.6504 17.5689 41.3561 17.653Z" fill="white"/><path d="M55.4217 17.8212H52.6465C52.5935 17.82 52.5414 17.8079 52.4932 17.7858C52.4451 17.7637 52.402 17.7319 52.3666 17.6926C52.3311 17.6532 52.3041 17.607 52.2872 17.5568C52.2702 17.5066 52.2637 17.4535 52.268 17.4007V12.2917H47.7056V17.4007C47.7085 17.4549 47.7002 17.5091 47.6815 17.5601C47.6627 17.611 47.6337 17.6575 47.5963 17.6969C47.5589 17.7362 47.5139 17.7676 47.464 17.7889C47.4141 17.8103 47.3604 17.8213 47.3061 17.8212H44.5308C44.4193 17.8212 44.3124 17.7769 44.2335 17.6981C44.1547 17.6192 44.1104 17.5123 44.1104 17.4007V6.67804C44.1104 2.88305 46.4021 2.04205 49.2614 2.04205H50.7963C53.6136 2.04205 55.8843 2.88305 55.8843 6.67804V17.4007C55.8846 17.4597 55.8725 17.518 55.8488 17.572C55.8251 17.626 55.7903 17.6743 55.7467 17.714C55.703 17.7536 55.6516 17.7837 55.5956 17.8021C55.5396 17.8206 55.4804 17.8271 55.4217 17.8212ZM52.268 6.67804C52.268 5.52167 51.9316 5.16425 50.7542 5.16425H49.2614C48.1051 5.16425 47.7477 5.52167 47.7477 6.67804V9.32717H52.3101L52.268 6.67804Z" fill="white"/><path d="M71.0851 2.63076L67.9313 15.6767C67.4478 17.6951 65.9129 18.0105 64.4622 18.0105C63.0115 18.0105 61.4557 17.6951 60.9721 15.6767L57.8184 2.63076C57.8184 2.63076 57.8184 2.56768 57.8184 2.52563C57.821 2.47867 57.833 2.43269 57.8535 2.39035C57.8739 2.348 57.9026 2.31012 57.9377 2.27886C57.9729 2.24761 58.0139 2.22361 58.0583 2.20822C58.1028 2.19284 58.1499 2.18639 58.1968 2.18923H61.1403C61.243 2.19507 61.3406 2.23574 61.4171 2.30455C61.4935 2.37335 61.5442 2.46617 61.5608 2.56768L64.1889 14.7096C64.1889 14.9408 64.294 15.0249 64.4832 15.0249C64.6725 15.0249 64.7355 14.9408 64.7776 14.7096L67.4057 2.56768C67.4177 2.46793 67.4642 2.37551 67.5372 2.30641C67.6101 2.23731 67.7049 2.19586 67.8052 2.18923H70.7487C70.7949 2.1847 70.8416 2.19013 70.8856 2.20513C70.9296 2.22014 70.9699 2.2444 71.0038 2.27626C71.0376 2.30812 71.0643 2.34686 71.0819 2.38987C71.0996 2.43288 71.1078 2.47917 71.1061 2.52563C71.097 2.56023 71.09 2.59533 71.0851 2.63076Z" fill="white"/><path d="M82.2914 17.7372C80.9416 17.9404 79.5774 18.0318 78.2126 18.0105C75.5004 18.0105 73.1035 17.3167 73.1035 13.4901V6.50987C73.1035 2.67283 75.5214 2.00003 78.2336 2.00003C79.5908 1.97134 80.9481 2.05573 82.2914 2.25233C82.5857 2.25233 82.7119 2.3995 82.7119 2.67283V4.65968C82.7092 4.77036 82.6641 4.87577 82.5858 4.95406C82.5075 5.03235 82.4021 5.07751 82.2914 5.08017H78.0233C77.0352 5.08017 76.6777 5.40606 76.6777 6.50987V8.42313H82.1232C82.2347 8.42313 82.3417 8.46743 82.4205 8.54629C82.4994 8.62515 82.5437 8.73211 82.5437 8.84363V10.841C82.5437 10.9525 82.4994 11.0595 82.4205 11.1383C82.3417 11.2172 82.2347 11.2615 82.1232 11.2615H76.6777V13.4901C76.6777 14.5414 77.0352 14.9198 78.0233 14.9198H82.2914C82.4029 14.9198 82.5099 14.9641 82.5887 15.043C82.6676 15.1218 82.7119 15.2288 82.7119 15.3403V17.2956C82.7119 17.569 82.5857 17.6951 82.2914 17.7372Z" fill="white"/><path d="M90.4494 18.021C88.951 18.0224 87.4545 17.917 85.9711 17.7056C85.8804 17.7038 85.7912 17.6826 85.7094 17.6432C85.6277 17.6038 85.5555 17.5472 85.4976 17.4774C85.4397 17.4076 85.3976 17.326 85.3741 17.2384C85.3506 17.1508 85.3462 17.0592 85.3614 16.9697V3.05126C85.3462 2.96182 85.3506 2.87017 85.3741 2.78256C85.3976 2.69495 85.4397 2.61344 85.4976 2.54359C85.5555 2.47374 85.6277 2.4172 85.7094 2.37781C85.7912 2.33843 85.8804 2.31714 85.9711 2.31539C87.4545 2.10401 88.951 1.99862 90.4494 2.00001C94.6544 2.00001 97.1563 4.20763 97.1563 8.65439V11.3666C97.1563 15.8028 94.6334 18.021 90.4494 18.021ZM93.6031 8.6649C93.6031 5.82654 92.3837 5.06965 90.4494 5.06965C89.9238 5.06965 89.251 5.06965 88.9566 5.06965V14.8883C89.251 14.8883 89.9238 14.8883 90.4494 14.8883C92.3416 14.8883 93.6031 14.1314 93.6031 11.293V8.6649Z" fill="white"/><path d="M110.549 17.6531C109.708 17.7792 107.08 18.0105 105.966 18.0105C102.244 18.0105 99.6582 16.7911 99.6582 11.9974V2.58869C99.6581 2.53441 99.6691 2.48069 99.6905 2.4308C99.7119 2.38091 99.7432 2.3359 99.7825 2.29852C99.8219 2.26113 99.8684 2.23216 99.9194 2.21337C99.9703 2.19458 100.024 2.18636 100.079 2.18921H102.875C102.929 2.18636 102.983 2.19458 103.034 2.21337C103.085 2.23216 103.132 2.26113 103.171 2.29852C103.211 2.3359 103.242 2.38091 103.263 2.4308C103.285 2.48069 103.296 2.53441 103.296 2.58869V12.0499C103.296 14.1524 103.863 15.0565 105.755 15.0565C106.578 15.0353 107.399 14.9722 108.215 14.8673V2.58869C108.215 2.53441 108.226 2.48069 108.248 2.4308C108.269 2.38091 108.3 2.3359 108.34 2.29852C108.379 2.26113 108.426 2.23216 108.477 2.21337C108.527 2.19458 108.582 2.18636 108.636 2.18921H111.369C111.424 2.18638 111.48 2.19448 111.532 2.21303C111.584 2.23159 111.632 2.26024 111.673 2.29735C111.714 2.33445 111.747 2.37929 111.771 2.42928C111.794 2.47927 111.808 2.53344 111.811 2.58869V16.2549C111.811 17.2326 111.432 17.5269 110.549 17.6531Z" fill="white"/><path d="M123.753 17.6531C122.722 17.9122 121.662 18.0324 120.599 18.0105C117.446 18.0105 114.376 16.3285 114.376 11.3246V8.67544C114.376 3.66099 117.404 2.00002 120.599 2.00002C121.662 1.97818 122.722 2.09839 123.753 2.35744C124.047 2.44154 124.216 2.52564 124.216 2.81999V4.85941C124.224 4.91707 124.219 4.97591 124.201 5.03136C124.183 5.08681 124.153 5.13736 124.112 5.17908C124.071 5.22079 124.022 5.25255 123.967 5.27186C123.912 5.29117 123.853 5.29753 123.795 5.29042C122.996 5.22735 121.84 5.12222 120.641 5.12222C119.191 5.12222 118.034 5.87912 118.034 8.67544V11.3246C118.034 14.1209 119.191 14.8778 120.641 14.8778C121.798 14.8778 122.954 14.7727 123.795 14.7096C123.852 14.7028 123.91 14.7091 123.964 14.7278C124.018 14.7466 124.067 14.7773 124.107 14.8178C124.148 14.8583 124.179 14.9074 124.197 14.9615C124.216 15.0156 124.222 15.0733 124.216 15.1301V17.1695C124.216 17.4639 124.047 17.569 123.753 17.6531Z" fill="white"/><path d="M138.681 2.71485L133.677 9.83181L138.702 17.3798C138.754 17.4457 138.784 17.5269 138.786 17.611C138.786 17.7372 138.681 17.8213 138.491 17.8213H134.875C134.778 17.8234 134.682 17.7965 134.599 17.7442C134.517 17.6918 134.452 17.6162 134.413 17.5269L130.145 10.5887V17.4008C130.145 17.5123 130.1 17.6193 130.021 17.6981C129.943 17.777 129.836 17.8213 129.724 17.8213H126.97C126.858 17.8213 126.751 17.777 126.672 17.6981C126.594 17.6193 126.549 17.5123 126.549 17.4008V2.5887C126.549 2.53443 126.56 2.4807 126.582 2.43082C126.603 2.38093 126.634 2.33592 126.674 2.29853C126.713 2.26115 126.76 2.23218 126.81 2.21338C126.861 2.19459 126.916 2.18637 126.97 2.18923H129.724C129.778 2.18637 129.833 2.19459 129.883 2.21338C129.934 2.23218 129.981 2.26115 130.02 2.29853C130.06 2.33592 130.091 2.38093 130.112 2.43082C130.134 2.4807 130.145 2.53443 130.145 2.5887V9.28516L134.77 2.44153C134.815 2.35925 134.882 2.29163 134.964 2.24685C135.046 2.20207 135.139 2.18206 135.233 2.18923H138.449C138.681 2.18923 138.786 2.29436 138.786 2.4205C138.776 2.52589 138.74 2.62711 138.681 2.71485Z" fill="white"/></g><path d="M22.0122 7.33333C26.1178 14.4444 24.065 18 15.8538 18C7.64258 18 5.58978 14.4444 9.69538 7.33333C13.801 0.222222 17.9066 0.222222 22.0122 7.33333Z" fill="white"/><path d="M12.8974 12.5956C14.8681 16.0089 13.8828 17.7156 9.94138 17.7156C6 17.7156 5.01466 16.0089 6.98534 12.5956C8.95603 9.18225 10.9267 9.18225 12.8974 12.5956Z" fill="#FFB400"/><path d="M16.5925 10.4623C17.0852 11.3157 16.8389 11.7423 15.8535 11.7423C14.8682 11.7423 14.6218 11.3157 15.1145 10.4623C15.6072 9.60899 16.0998 9.60899 16.5925 10.4623Z" fill="black"/><defs><clipPath id="clip0_13_533"><rect width="106.786" height="16" fill="white" transform="translate(32 2)"/></clipPath></defs></svg>
                            </div>
                            <div style="display: flex; justify-content: center; padding:8px 0px; align-items:baseline;">
                                <a href="https://github.com/modulabs-personalab/psyctl" style="display: flex;align-items: end; padding:0;">
                                <img style="max-width: 24px;" src="https://github.com/modulabs-personalab/psyctl/raw/main/docs/images/logo.png"> <span style="padding-bottom:2px;">Git repository</span>
                                </a>
                            </div>
                        </div>
                    """)
            

                
                    

        # 이벤트 핸들러 연결

        # 벡터 선택 시 정보 표시
        vector_dropdown.change(
            fn=get_model_info_for_vector,
            inputs=[vector_dropdown],
            outputs=[vector_info_text],
        )

        # 스티어링 적용 버튼 (generator로 실시간 상태 업데이트)
        apply_steering_btn.click(
            fn=change_steering_vector_generator,
            inputs=[vector_dropdown, strength_slider, force_cpu_checkbox, orthogonal_checkbox],
            outputs=[steering_status],
        )

        # 강도 슬라이더 변경 시
        strength_slider.release(
            fn=update_steering_params,
            inputs=[strength_slider, orthogonal_checkbox],
            outputs=[steering_status],
        )

        # Orthogonal 체크박스 변경 시
        orthogonal_checkbox.change(
            fn=update_steering_params,
            inputs=[strength_slider, orthogonal_checkbox],
            outputs=[steering_status],
        )

        def user_message(message, chat_history):
            """사용자 메시지 추가"""
            if not message.strip():
                return "", chat_history
            # ChatMessage 사용
            return "", chat_history + [ChatMessage(role="user", content=message)]

        def bot_response(chat_history, temperature, max_tokens):
            """봇 응답 스트리밍 생성"""
            import datetime
            print(f"[BOT_DEBUG] bot_response 시작, chat_history_len={len(chat_history) if chat_history else 0}")
            if not chat_history:
                print("[BOT_DEBUG] chat_history 비어있음")
                yield chat_history, gr.update()
                return

            # 마지막 사용자 메시지 가져오기
            last_msg = chat_history[-1]
            if hasattr(last_msg, "content"):
                user_msg = extract_text_content(last_msg.content)
            else:
                user_msg = extract_text_content(last_msg.get("content", ""))

            print(f"[BOT_DEBUG] user_msg: '{user_msg[:50] if user_msg else ''}...'")

            # 이전 히스토리 (마지막 사용자 메시지 제외)
            prev_history = chat_history[:-1]

            # 스트리밍 시작 로그
            start_time = datetime.datetime.now()
            start_log = f"[{start_time.strftime('%H:%M:%S')}] 스트리밍 시작..."

            # 스트리밍 응답 생성
            response_text = ""
            yield_count = 0
            for partial_response in chat_stream(user_msg, prev_history, temperature, max_tokens):
                response_text = partial_response
                yield_count += 1
                if yield_count <= 3 or yield_count % 10 == 0:
                    print(f"[BOT_DEBUG] yield {yield_count}: response_len={len(response_text)}")
                # 새로운 ChatMessage로 히스토리 구성
                yield chat_history + [ChatMessage(role="assistant", content=response_text)], start_log

            # 스트리밍 종료 로그
            end_time = datetime.datetime.now()
            elapsed = (end_time - start_time).total_seconds()
            end_log = f"[{start_time.strftime('%H:%M:%S')}] 스트리밍 시작\n[{end_time.strftime('%H:%M:%S')}] 스트리밍 종료 (소요: {elapsed:.1f}초, 토큰: {yield_count}개)"
            yield chat_history + [ChatMessage(role="assistant", content=response_text)], end_log

            print(f"[BOT_DEBUG] bot_response 종료, 총 yield: {yield_count}, 최종 응답 길이: {len(response_text)}")

        # 전송 버튼: 먼저 사용자 메시지 추가, 그 다음 봇 응답 생성
        send_btn.click(
            fn=user_message,
            inputs=[msg_input, chatbot],
            outputs=[msg_input, chatbot],
            queue=False,
        ).then(
            fn=bot_response,
            inputs=[chatbot, temperature_slider, max_tokens_slider],
            outputs=[chatbot, steering_status],
        )

        # 엔터키로 전송
        msg_input.submit(
            fn=user_message,
            inputs=[msg_input, chatbot],
            outputs=[msg_input, chatbot],
            queue=False,
        ).then(
            fn=bot_response,
            inputs=[chatbot, temperature_slider, max_tokens_slider],
            outputs=[chatbot, steering_status],
        )

        clear_btn.click(
            fn=lambda: [],
            outputs=[chatbot],
            queue=False,
        )

    return demo


def main():
    """메인 함수"""
    print("[DEBUG] main() 시작", flush=True)

    print("[DEBUG] initialize_app() 호출 전", flush=True)
    initialize_app()
    print("[DEBUG] initialize_app() 완료", flush=True)

    print("[DEBUG] create_ui() 호출 전", flush=True)
    demo = create_ui()
    print("[DEBUG] create_ui() 완료", flush=True)

    print("[DEBUG] demo.launch() 호출 전", flush=True)
    print(f"[DEBUG] Gradio version: {gr.__version__}", flush=True)
    print(f"[DEBUG] 서버 바인딩 시도: 0.0.0.0:{GRADIO_PORT}", flush=True)
    sys.stdout.flush()
    sys.stderr.flush()

    # 포트 사용 가능 여부 확인
    import socket
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        result = sock.connect_ex(('127.0.0.1', GRADIO_PORT))
        if result == 0:
            print(f"[DEBUG] 포트 {GRADIO_PORT} 이미 사용 중!", flush=True)
        else:
            print(f"[DEBUG] 포트 {GRADIO_PORT} 사용 가능", flush=True)
        sock.close()
    except Exception as e:
        print(f"[DEBUG] 포트 확인 실패: {e}", flush=True)

    print("[DEBUG] demo.launch() 실행 중...", flush=True)

    # Gradio analytics 비활성화
    os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"

    demo.launch(
        server_name="0.0.0.0",  # 외부 접근 허용 (런팟용)
        server_port=GRADIO_PORT,
        share=False,  # 공개 링크 생성 여부
        show_error=True,
    )
    print("[DEBUG] demo.launch() 완료", flush=True)


if __name__ == "__main__":
    main()
