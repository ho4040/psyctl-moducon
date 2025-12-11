"""
HuggingFace에서 스티어링 벡터 다운로드 및 관리
"""

from pathlib import Path

from huggingface_hub import hf_hub_download, list_repo_files

# 기본 스티어링 벡터 저장소
DEFAULT_REPO = "dalekwon/bipo-steering-vectors"

# 사용 가능한 스티어링 벡터 목록
# 구조: {표시 이름: {"filename": 파일명, "model": 모델명, "layer": 레이어 경로}}
AVAILABLE_VECTORS = {
    "English - Agreeableness (친화성)": {
        "filename": "bipo_steering_english_agreeableness.safetensors",
        "model": "meta-llama/Llama-3.1-8B-Instruct",
        "layer": "model.layers.13.mlp",
    },
    "English - Awfully Sweet (지나치게 다정함)": {
        "filename": "bipo_steering_english_awfully_sweet.safetensors",
        "model": "meta-llama/Llama-3.1-8B-Instruct",
        "layer": "model.layers.13.mlp",
    },
    "English - Neuroticism (신경증)": {
        "filename": "bipo_steering_english_neuroticism.safetensors",
        "model": "meta-llama/Llama-3.1-8B-Instruct",
        "layer": "model.layers.13.mlp",
    },
    "English - Paranoid (편집증)": {
        "filename": "bipo_steering_english_paranoid.safetensors",
        "model": "meta-llama/Llama-3.1-8B-Instruct",
        "layer": "model.layers.13.mlp",
    },
    "Korean - Awfully Sweet (지나치게 다정함)": {
        "filename": "bipo_steering_korean_awfully_sweet.safetensors",
        "model": "LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct",
        "layer": "transformer.h.13.mlp",
    },
    "Korean - Rude (무례함)": {
        "filename": "bipo_steering_korean_rude.safetensors",
        "model": "LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct",
        "layer": "transformer.h.13.mlp",
    },
}


class VectorManager:
    """스티어링 벡터 다운로드 및 관리 클래스"""

    def __init__(
        self,
        cache_dir: Path | str = "./vectors",
        repo_id: str = DEFAULT_REPO,
    ):
        """
        Args:
            cache_dir: 벡터 파일 캐시 디렉토리
            repo_id: HuggingFace 저장소 ID
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.repo_id = repo_id

    def list_available_vectors(self) -> dict[str, str]:
        """
        사용 가능한 스티어링 벡터 목록 반환

        Returns:
            {표시 이름: 파일 이름} 딕셔너리
        """
        return AVAILABLE_VECTORS.copy()

    def list_remote_vectors(self) -> list[str]:
        """
        HuggingFace 저장소의 벡터 파일 목록 조회

        Returns:
            safetensors 파일 이름 목록
        """
        try:
            files = list_repo_files(self.repo_id)
            return [f for f in files if f.endswith(".safetensors")]
        except Exception as e:
            print(f"[WARNING] 원격 저장소 조회 실패: {e}")
            return list(AVAILABLE_VECTORS.values())

    def download_vector(self, filename: str) -> Path:
        """
        스티어링 벡터 파일 다운로드

        Args:
            filename: 다운로드할 파일 이름

        Returns:
            다운로드된 파일 경로
        """
        local_path = self.cache_dir / filename

        # 이미 다운로드되어 있으면 캐시 사용
        if local_path.exists():
            print(f"[INFO] 캐시 사용: {local_path}")
            return local_path

        print(f"[INFO] 다운로드 중: {filename}")
        downloaded_path = hf_hub_download(
            repo_id=self.repo_id,
            filename=filename,
            local_dir=self.cache_dir,
            local_dir_use_symlinks=False,
        )
        print(f"[INFO] 다운로드 완료: {downloaded_path}")

        return Path(downloaded_path)

    def get_vector_path(self, display_name: str) -> Path | None:
        """
        표시 이름으로 벡터 파일 경로 가져오기 (필요시 다운로드)

        Args:
            display_name: UI에 표시되는 벡터 이름

        Returns:
            벡터 파일 경로 (없으면 None)
        """
        vector_info = AVAILABLE_VECTORS.get(display_name)
        if vector_info is None:
            print(f"[ERROR] 알 수 없는 벡터: {display_name}")
            return None

        return self.download_vector(vector_info["filename"])

    def get_vector_info(self, display_name: str) -> dict | None:
        """
        벡터의 전체 정보 가져오기

        Args:
            display_name: UI에 표시되는 벡터 이름

        Returns:
            벡터 정보 딕셔너리 (filename, model, layer) 또는 None
        """
        return AVAILABLE_VECTORS.get(display_name)

    def get_model_for_vector(self, display_name: str) -> str | None:
        """
        벡터에 맞는 모델 이름 가져오기

        Args:
            display_name: UI에 표시되는 벡터 이름

        Returns:
            모델 이름 또는 None
        """
        vector_info = AVAILABLE_VECTORS.get(display_name)
        if vector_info is None:
            return None
        return vector_info["model"]

    def download_all_vectors(self) -> list[Path]:
        """
        모든 스티어링 벡터 다운로드

        Returns:
            다운로드된 파일 경로 목록
        """
        paths = []
        for vector_info in AVAILABLE_VECTORS.values():
            path = self.download_vector(vector_info["filename"])
            paths.append(path)
        return paths
