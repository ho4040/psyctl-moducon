"""
스티어링 벡터 로드 및 모델에 적용하는 모듈
psyctl 프로젝트의 steering_applier.py를 참고하여 구현
"""

import ast
import json
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from safetensors import safe_open
from safetensors.torch import load_file


class SteeringVector:
    """스티어링 벡터를 로드하고 관리하는 클래스"""

    def __init__(self, filepath: Path | str):
        """
        safetensors 파일에서 스티어링 벡터 로드

        Args:
            filepath: 스티어링 벡터 파일 경로
        """
        self.filepath = Path(filepath)
        self.vectors: dict[str, torch.Tensor] = {}
        self.metadata: dict[str, Any] = {}
        self._load()

    def _load(self) -> None:
        """safetensors 파일에서 벡터와 메타데이터 로드"""
        # 텐서 로드
        data = load_file(self.filepath)

        # 메타데이터 로드
        with safe_open(self.filepath, framework="pt") as f:
            metadata_raw = f.metadata()
            if metadata_raw:
                self.metadata = {k: v for k, v in metadata_raw.items()}

        # layer_names 파싱 (Python repr 또는 JSON 문자열로 저장됨)
        layer_names_str = self.metadata.get("layer_names", "[]")
        try:
            # ast.literal_eval로 Python repr 문자열 파싱 시도
            layer_names = ast.literal_eval(layer_names_str)
            if isinstance(layer_names, str):
                layer_names = [layer_names]
        except (ValueError, SyntaxError):
            try:
                # JSON 파싱 시도
                layer_names = json.loads(layer_names_str)
            except json.JSONDecodeError:
                # 단일 레이어인 경우
                layer_names = [layer_names_str] if layer_names_str else []

        # 벡터 딕셔너리 구성
        if layer_names:
            for idx, layer_name in enumerate(layer_names):
                tensor_key = f"layer_{idx}"
                if tensor_key in data:
                    self.vectors[layer_name] = data[tensor_key]
        else:
            # layer_names가 없으면 데이터 키를 그대로 사용
            self.vectors = data

    @property
    def model_name(self) -> str:
        """스티어링 벡터가 만들어진 모델 이름"""
        return self.metadata.get("model", "unknown")

    @property
    def layer_names(self) -> list[str]:
        """스티어링 벡터가 적용될 레이어 이름 목록"""
        return list(self.vectors.keys())


class SteeringManager:
    """모델에 스티어링 벡터를 적용하고 관리하는 클래스"""

    def __init__(self):
        self._hooks: list[torch.utils.hooks.RemovableHandle] = []
        self._current_vector: SteeringVector | None = None
        self._current_strength: float = 1.0
        self._current_orthogonal: bool = False
        self._current_model: nn.Module | None = None

    def _get_layer_module(self, model: nn.Module, layer_path: str) -> nn.Module:
        """
        레이어 경로 문자열로 모델의 특정 레이어 모듈 가져오기

        Args:
            model: 대상 모델
            layer_path: 레이어 경로 (예: "model.layers[13].mlp.down_proj")

        Returns:
            해당 레이어 모듈
        """
        # 경로 파싱: model.layers[13].mlp.down_proj -> ["model", "layers", "13", "mlp", "down_proj"]
        import re

        components = re.split(r"\.|\[|\]", layer_path)
        components = [c for c in components if c]  # 빈 문자열 제거

        current = model
        for component in components:
            if component.isdigit():
                current = current[int(component)]
            else:
                current = getattr(current, component)

        return current

    def _make_steering_hook(
        self,
        steer_vec: torch.Tensor,
        strength: float,
        prompt_length: int = 0,
        orthogonal: bool = False,
    ):
        """
        스티어링을 적용하는 forward hook 생성

        Args:
            steer_vec: 스티어링 벡터
            strength: 스티어링 강도 (-5 ~ 5)
            prompt_length: 프롬프트 토큰 길이 (이후 토큰에만 적용)
            orthogonal: True일 경우 직교 방식으로 스티어링 적용
        """

        def hook(module, input, output):
            # 출력 처리 (tuple vs tensor)
            if isinstance(output, tuple):
                out = output[0]
                extra_outputs = output[1:]
            else:
                out = output
                extra_outputs = ()

            # 복제 및 부동소수점 변환
            if not torch.is_floating_point(out):
                out = out.float()
            out = out.clone()

            # 스티어링 벡터를 모델 device/dtype에 맞추기
            steer = steer_vec.to(device=out.device, dtype=out.dtype)
            steer_reshaped = steer.view(1, 1, -1)  # [1, 1, hidden_dim]

            # 프롬프트 이후 토큰에만 스티어링 적용
            if prompt_length > 0 and out.shape[1] > prompt_length:
                if orthogonal:
                    # Orthogonalized addition: 기존 성분 제거 후 스티어링 추가
                    norm_steer = steer / (steer.norm(p=2) + 1e-8)
                    norm_steer_reshaped = norm_steer.view(1, 1, -1)
                    proj_coeff = (out[:, prompt_length:, :] * norm_steer_reshaped).sum(
                        dim=-1, keepdim=True
                    )
                    proj = proj_coeff * norm_steer_reshaped
                    out[:, prompt_length:, :] = (
                        out[:, prompt_length:, :] - proj
                    ) + strength * steer_reshaped
                else:
                    # Simple addition
                    out[:, prompt_length:, :] = out[:, prompt_length:, :] + strength * steer_reshaped
            else:
                # 전체 시퀀스에 적용
                if orthogonal:
                    norm_steer = steer / (steer.norm(p=2) + 1e-8)
                    norm_steer_reshaped = norm_steer.view(1, 1, -1)
                    proj_coeff = (out * norm_steer_reshaped).sum(dim=-1, keepdim=True)
                    proj = proj_coeff * norm_steer_reshaped
                    out = (out - proj) + strength * steer_reshaped
                else:
                    out = out + strength * steer_reshaped

            if extra_outputs:
                return (out, *extra_outputs)
            return out

        return hook

    def apply_steering(
        self,
        model: nn.Module,
        steering_vector: SteeringVector,
        strength: float = 1.0,
        prompt_length: int = 0,
        orthogonal: bool = False,
    ) -> None:
        """
        모델에 스티어링 벡터 적용

        Args:
            model: 대상 모델
            steering_vector: 적용할 스티어링 벡터
            strength: 스티어링 강도 (-5 ~ 5)
            prompt_length: 프롬프트 토큰 길이
            orthogonal: True일 경우 직교 방식으로 스티어링 적용
        """
        # 기존 훅 제거
        self.remove_steering()

        self._current_model = model
        self._current_vector = steering_vector
        self._current_strength = strength
        self._current_orthogonal = orthogonal

        # 각 레이어에 훅 등록
        for layer_name, steer_vec in steering_vector.vectors.items():
            try:
                layer_module = self._get_layer_module(model, layer_name)
                hook = self._make_steering_hook(steer_vec, strength, prompt_length, orthogonal)
                handle = layer_module.register_forward_hook(hook)
                self._hooks.append(handle)
            except (AttributeError, IndexError) as e:
                print(f"[WARNING] 레이어 '{layer_name}'를 찾을 수 없음: {e}")

    def remove_steering(self) -> None:
        """모든 스티어링 훅 제거"""
        for handle in self._hooks:
            handle.remove()
        self._hooks.clear()
        self._current_vector = None
        self._current_strength = 1.0
        self._current_orthogonal = False
        self._current_model = None

    def unload_all(self) -> None:
        """
        모델과 스티어링 벡터 모두 메모리에서 정리

        훅 제거 + 스티어링 벡터 참조 해제 + CUDA 캐시 정리
        """
        # 훅 먼저 제거
        self.remove_steering()

        # CUDA 메모리 정리
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        print("[INFO] SteeringManager: 모든 리소스 정리 완료")

    def update_strength(
        self,
        model: nn.Module,
        strength: float,
        prompt_length: int = 0,
        orthogonal: bool | None = None,
    ) -> None:
        """
        스티어링 강도만 업데이트

        Args:
            model: 대상 모델
            strength: 새로운 스티어링 강도
            prompt_length: 프롬프트 토큰 길이
            orthogonal: True/False로 지정하면 해당 값 사용, None이면 기존 값 유지
        """
        if self._current_vector is not None:
            orth = orthogonal if orthogonal is not None else self._current_orthogonal
            self.apply_steering(model, self._current_vector, strength, prompt_length, orth)

    @property
    def is_active(self) -> bool:
        """스티어링이 활성화되어 있는지 여부"""
        return len(self._hooks) > 0
