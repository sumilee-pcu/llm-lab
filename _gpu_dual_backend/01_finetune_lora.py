"""
GPU 구간 — 파인튜닝 / PEFT (LoRA)   [Windows · macOS 듀얼 백엔드]
강의 요소: 사전학습 모델을 적은 파라미터(LoRA)만 학습해 도메인/말투에 맞춤

【교재 → 2026 업데이트】
  - HF Trainer + peft (GPU/Colab 전제)  →  환경 자동 감지 듀얼 백엔드
      · Apple Silicon : mlx-lm 으로 칩 내장 GPU에서 네이티브 학습
      · Windows/Colab : transformers + peft (CUDA 있으면 GPU, 없으면 CPU 초소형 모델)
  - 두 경로 모두 'LoRA 어댑터 저장 + loss 감소'라는 동일 결과를 만든다.

실행:  python 01_finetune_lora.py
사전설치:  pip install -r requirements-gpu.txt   (환경에 맞는 줄만 설치됨)
"""
import os
import subprocess
import sys
from backend import print_env

BACKEND = print_env()
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")


def run_mlx():
    """Apple Silicon: mlx-lm LoRA. 초소형 0.5B-4bit 모델로 몇 초만에 완료."""
    print("\n── MLX LoRA 파인튜닝 시작 ──")
    cmd = [
        sys.executable, "-m", "mlx_lm", "lora",
        "--model", "mlx-community/Qwen2.5-0.5B-Instruct-4bit",
        "--train", "--data", DATA_DIR,
        "--fine-tune-type", "lora",
        "--batch-size", "1", "--num-layers", "4", "--iters", "30",
        "--steps-per-report", "10", "--steps-per-eval", "30",
        "--adapter-path", os.path.join(os.path.dirname(__file__), "adapters_mlx"),
    ]
    subprocess.run(cmd, check=True)
    print("✅ MLX: adapters_mlx/adapters.safetensors 저장 완료")


def run_torch():
    """Windows/Colab/CPU: transformers + peft. CUDA 있으면 GPU, 없으면 CPU."""
    import json
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig, get_peft_model
    from backend import torch_device

    device = torch_device()
    print(f"\n── torch LoRA 파인튜닝 시작 (device={device}) ──")

    # CPU에서도 끝까지 돌도록 초소형 모델. Colab GPU면 더 큰 모델로 교체 가능.
    model_id = "sshleifer/tiny-gpt2"
    tok = AutoTokenizer.from_pretrained(model_id)
    tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_id).to(device)

    lora = LoraConfig(r=4, lora_alpha=8, target_modules=["c_attn"], task_type="CAUSAL_LM")
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()

    # data/train.jsonl 의 prompt+completion 을 학습 텍스트로 사용
    texts = []
    with open(os.path.join(DATA_DIR, "train.jsonl"), encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            texts.append(f"{r['prompt']} {r['completion']}")

    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    model.train()
    for step in range(30):
        enc = tok(texts, return_tensors="pt", padding=True, truncation=True, max_length=64).to(device)
        out = model(**enc, labels=enc["input_ids"])
        out.loss.backward(); opt.step(); opt.zero_grad()
        if step % 10 == 0 or step == 29:
            print(f"Iter {step}: loss {out.loss.item():.3f}")

    out_dir = os.path.join(os.path.dirname(__file__), "adapters_torch")
    model.save_pretrained(out_dir)
    print(f"✅ torch: {out_dir}/ 어댑터 저장 완료")


if BACKEND == "mlx":
    run_mlx()
else:
    run_torch()

print("\n🎯 결론: 동일한 LoRA 개념을 두 환경 모두에서 '실제 학습'으로 시연 완료")
