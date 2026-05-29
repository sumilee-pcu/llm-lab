"""
GPU 구간 — 양자화 (Quantization)   [Windows · macOS 듀얼 백엔드]
강의 요소: 모델 가중치를 저비트(4/8bit)로 줄여 메모리·용량 절감

【교재 → 2026 업데이트】
  - bitsandbytes (CUDA 전용, Windows/CPU 제약 많음)
      · Apple Silicon : mlx_lm.convert -q  (4bit/8bit, affine·mxfp4 등 네이티브)
      · Windows/Colab : transformers BitsAndBytesConfig (CUDA) / 개념 시연(CPU)
  - 핵심 메시지는 동일: "같은 모델, 더 작은 메모리".

실행:  python 02_quantize.py
"""
import os
import subprocess
import sys
from backend import print_env

BACKEND = print_env()


def run_mlx():
    print("\n── MLX 4bit 양자화 시작 ──")
    out = os.path.join(os.path.dirname(__file__), "qmodel_mlx")
    cmd = [
        sys.executable, "-m", "mlx_lm", "convert",
        "--hf-path", "Qwen/Qwen2.5-0.5B-Instruct",
        "--mlx-path", out,
        "-q", "--q-bits", "4", "--q-group-size", "64",
    ]
    subprocess.run(cmd, check=True)
    print(f"✅ MLX: {out}/ 에 4bit 양자화 모델 저장 (용량/메모리 대폭 감소)")


def run_torch():
    import torch
    from backend import torch_device
    device = torch_device()
    if device == "cuda":
        print("\n── torch: BitsAndBytesConfig 4bit 로드 (CUDA) ──")
        from transformers import AutoModelForCausalLM, BitsAndBytesConfig
        cfg = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
        m = AutoModelForCausalLM.from_pretrained(
            "sshleifer/tiny-gpt2", quantization_config=cfg, device_map="auto")
        print("✅ torch(CUDA): 4bit 양자화 로드 완료")
    else:
        # CPU/MPS: bitsandbytes 4bit 미지원 → 개념 시연(동적 양자화)으로 대체
        print("\n── torch(CPU): 동적 양자화 개념 시연 ──")
        from transformers import AutoModelForCausalLM
        m = AutoModelForCausalLM.from_pretrained("sshleifer/tiny-gpt2")
        before = sum(p.numel() * p.element_size() for p in m.parameters())
        qm = torch.quantization.quantize_dynamic(m, {torch.nn.Linear}, dtype=torch.qint8)
        print(f"✅ torch(CPU): 동적 8bit 양자화 적용 (Linear 레이어) — 원본 가중치 {before/1e3:.1f}KB")
        print("   ⚠️ 풀 4bit(bitsandbytes)는 CUDA 필요 → Colab 또는 Apple Silicon(MLX) 권장")


if BACKEND == "mlx":
    run_mlx()
else:
    run_torch()

print("\n🎯 결론: 양자화의 '메모리 절감' 개념을 두 환경 모두에서 시연 완료")
