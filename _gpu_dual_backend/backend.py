"""
GPU 구간 듀얼 백엔드 디스패처 (Windows · macOS 공용)
====================================================

이 모듈 하나만 import 하면, 실행 환경을 자동 감지해
- Apple Silicon(M-시리즈) + mlx 설치됨  → "mlx"  (칩 내장 GPU로 네이티브 학습)
- 그 외(Windows/Intel/Colab)            → "torch" (CUDA 있으면 GPU, 없으면 CPU)
중 하나를 골라줍니다.

강의 원칙
  · 같은 레슨, 같은 데이터, 같은 출력 지표 → 백엔드만 다름
  · 수강생은 자기 환경 그대로 실행하면 됨 (분기 코드를 직접 안 만져도 됨)
  · GPU가 꼭 필요한 무거운 주제는 '초소형 모델'로 CPU에서도 끝까지 돌게 구성
"""
import platform
import importlib.util


def has(pkg: str) -> bool:
    """패키지 설치 여부."""
    return importlib.util.find_spec(pkg) is not None


def detect_backend() -> str:
    """현재 환경에 맞는 백엔드 문자열을 반환한다 ('mlx' 또는 'torch')."""
    is_apple_silicon = platform.system() == "Darwin" and platform.machine() == "arm64"
    if is_apple_silicon and has("mlx"):
        return "mlx"
    return "torch"


def torch_device() -> str:
    """torch 경로에서 쓸 디바이스: cuda > mps > cpu."""
    import torch
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def print_env() -> str:
    """현재 환경을 한 줄로 출력하고 백엔드를 반환 (레슨 첫 줄에서 호출)."""
    be = detect_backend()
    sysname = f"{platform.system()} / {platform.machine()}"
    if be == "mlx":
        print(f"🖥  환경: {sysname}  →  백엔드: MLX (Apple Silicon 내장 GPU)")
    else:
        dev = torch_device() if has("torch") else "torch 미설치"
        print(f"🖥  환경: {sysname}  →  백엔드: torch ({dev})")
    return be


if __name__ == "__main__":
    print_env()
