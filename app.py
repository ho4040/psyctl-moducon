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
