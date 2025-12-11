"""
LLM 모델 로드 모듈
CPU/GPU 환경 자동 감지 및 메모리 최적화
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
from threading import Thread


def get_device() -> str:
    """
    사용 가능한 디바이스 반환

    Returns:
        "cuda" (GPU 사용 가능) 또는 "cpu"
    """
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def get_device_info() -> dict:
    """
    디바이스 정보 반환

    Returns:
        디바이스 정보 딕셔너리
    """
    device = get_device()
    info = {
        "device": device,
        "cuda_available": torch.cuda.is_available(),
    }

    if torch.cuda.is_available():
        info["gpu_name"] = torch.cuda.get_device_name(0)
        info["gpu_memory_total"] = torch.cuda.get_device_properties(0).total_memory / 1e9
        info["gpu_memory_allocated"] = torch.cuda.memory_allocated(0) / 1e9

    return info


class ModelLoader:
    """LLM 모델 로더 클래스"""

    def __init__(self, model_name: str = "meta-llama/Llama-3.1-8B-Instruct"):
        """
        Args:
            model_name: HuggingFace 모델 이름
        """
        self.model_name = model_name
        self.model = None
        self.tokenizer = None
        self.device = get_device()

    def load(self, force_cpu: bool = False) -> tuple:
        """
        모델과 토크나이저 로드

        Args:
            force_cpu: True면 GPU가 있어도 CPU 사용

        Returns:
            (model, tokenizer) 튜플
        """
        device = "cpu" if force_cpu else self.device

        print(f"[INFO] 모델 로드 중: {self.model_name}")
        print(f"[INFO] 디바이스: {device}")

        # 토크나이저 로드
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True,
        )

        # pad_token 설정 (없는 경우)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # 모델 로드 설정
        load_kwargs = {
            "trust_remote_code": True,
            "low_cpu_mem_usage": True,
        }

        if device == "cuda":
            # GPU: bfloat16 또는 float16 사용
            load_kwargs["torch_dtype"] = torch.bfloat16
            load_kwargs["device_map"] = "auto"
        else:
            # CPU: float32 사용, 메모리 절약을 위해 8bit 양자화 고려
            load_kwargs["torch_dtype"] = torch.float32

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            **load_kwargs,
        )

        # CPU인 경우 명시적으로 디바이스 이동
        if device == "cpu":
            self.model = self.model.to(device)

        self.model.eval()
        print(f"[INFO] 모델 로드 완료")

        return self.model, self.tokenizer

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
        do_sample: bool = True,
    ) -> str:
        """
        텍스트 생성

        Args:
            prompt: 입력 프롬프트
            max_new_tokens: 최대 생성 토큰 수
            temperature: 샘플링 온도
            top_p: nucleus sampling 파라미터
            top_k: top-k sampling 파라미터
            do_sample: 샘플링 사용 여부

        Returns:
            생성된 텍스트
        """
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("모델이 로드되지 않음. load()를 먼저 호출하세요.")

        # 입력 토큰화
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(self.model.device)

        # 생성
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                do_sample=do_sample,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                use_cache=True,
            )

        # 디코딩 (입력 제외)
        generated = self.tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
        )

        return generated

    def generate_stream(
        self,
        prompt: str,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
        do_sample: bool = True,
    ):
        """
        텍스트 스트리밍 생성 (generator)

        Args:
            prompt: 입력 프롬프트
            max_new_tokens: 최대 생성 토큰 수
            temperature: 샘플링 온도
            top_p: nucleus sampling 파라미터
            top_k: top-k sampling 파라미터
            do_sample: 샘플링 사용 여부

        Yields:
            생성된 텍스트 (누적)
        """
        print(f"[MODEL_DEBUG] generate_stream 진입")
        print(f"[MODEL_DEBUG] params: max_new_tokens={max_new_tokens}, temperature={temperature}, do_sample={do_sample}")

        if self.model is None or self.tokenizer is None:
            print("[MODEL_DEBUG] 모델 또는 토크나이저 없음")
            raise RuntimeError("모델이 로드되지 않음. load()를 먼저 호출하세요.")

        print(f"[MODEL_DEBUG] 모델 디바이스: {self.model.device}")

        # 입력 토큰화
        print(f"[MODEL_DEBUG] 프롬프트 토큰화 중... (프롬프트 길이: {len(prompt)})")
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(self.model.device)
        print(f"[MODEL_DEBUG] 입력 토큰 수: {inputs['input_ids'].shape[1]}")

        # 스트리머 설정
        print("[MODEL_DEBUG] TextIteratorStreamer 설정 중...")
        streamer = TextIteratorStreamer(
            self.tokenizer,
            skip_prompt=True,
            skip_special_tokens=True,
        )

        # 생성 kwargs
        generation_kwargs = {
            **inputs,
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "do_sample": do_sample,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
            "use_cache": True,
            "streamer": streamer,
        }
        print(f"[MODEL_DEBUG] generation_kwargs 설정 완료")

        # 별도 스레드에서 생성 실행
        print("[MODEL_DEBUG] 생성 스레드 시작...")
        thread = Thread(target=self.model.generate, kwargs=generation_kwargs)
        thread.start()
        print("[MODEL_DEBUG] 스레드 시작됨, 스트리밍 대기 중...")

        # 스트리밍 출력
        generated_text = ""
        chunk_count = 0
        try:
            for new_text in streamer:
                chunk_count += 1
                generated_text += new_text
                if chunk_count <= 3 or chunk_count % 10 == 0:
                    print(f"[MODEL_DEBUG] chunk {chunk_count}: +'{new_text}' (총 {len(generated_text)}자)")
                yield generated_text
        except Exception as e:
            print(f"[MODEL_DEBUG] 스트리밍 중 예외: {e}")
            raise

        print(f"[MODEL_DEBUG] 스트리밍 완료, 총 {chunk_count}개 청크, {len(generated_text)}자 생성")
        print("[MODEL_DEBUG] 스레드 join 대기...")
        thread.join()
        print("[MODEL_DEBUG] 스레드 종료됨")

    def get_prompt_length(self, prompt: str) -> int:
        """
        프롬프트의 토큰 길이 반환

        Args:
            prompt: 입력 프롬프트

        Returns:
            토큰 길이
        """
        if self.tokenizer is None:
            raise RuntimeError("토크나이저가 로드되지 않음")

        tokens = self.tokenizer(prompt, return_tensors="pt")
        return tokens["input_ids"].shape[1]

    def apply_chat_template(self, messages: list[dict]) -> str:
        """
        채팅 메시지에 템플릿 적용

        Args:
            messages: [{"role": "user", "content": "..."}, ...] 형태의 메시지 목록

        Returns:
            템플릿이 적용된 프롬프트 문자열
        """
        if self.tokenizer is None:
            raise RuntimeError("토크나이저가 로드되지 않음")

        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
