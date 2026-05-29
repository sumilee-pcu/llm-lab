"""
GPU 구간 — RLHF / 선호 정렬 (DPO)   [Windows · macOS 듀얼 백엔드]
해당 교재: 27 트랜스포머 아키텍처로 배우는 AI 에이전트 (RLHF·PPO·PEFT)

【교재 → 2026 업데이트 · Windows 경로 팩트체크】
  - 교재는 RLHF=PPO 위주. 2026 현실:
      · TRL v1.0(2026-03)에서 **PPO는 trl.experimental.ppo 로 이동** (안정 API 아님)
        → from trl.experimental.ppo import PPOTrainer, PPOConfig
      · 안정 트레이너는 SFT / **DPO** / Reward / RLOO / GRPO
      · 실무·강의 표준은 PPO 대신 **DPO**(보상모델·롤아웃 불필요, 선호쌍만으로 정렬)
  - 따라서 이 파일은 **DPO(안정 API)** 를 실제 실행하고, PPO는 개념+이동경로만 안내.
  - 디바이스 자동: Windows/Colab=cuda→cpu, macOS=mps. (소형 모델로 CPU도 동작)

실행:  python 06_rlhf_dpo.py
설치:  pip install "trl>=1.0" "transformers>=5.0" "peft>=0.18" datasets torch
"""
from backend import print_env, torch_device

print_env()

from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DPOConfig, DPOTrainer

DEVICE = torch_device()
print(f"학습 디바이스: {DEVICE}")

# 강의용 초소형 인과 LM
MODEL_ID = "HuggingFaceTB/SmolLM2-135M-Instruct"
tok = AutoTokenizer.from_pretrained(MODEL_ID)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
model = AutoModelForCausalLM.from_pretrained(MODEL_ID).to(DEVICE)

# 선호 데이터: 같은 prompt에 chosen(선호) / rejected(비선호) 쌍
pref = {
    "prompt": ["사용자에게 공손하게 인사해줘", "파이썬이 뭔지 한 줄로"],
    "chosen": ["안녕하세요! 무엇을 도와드릴까요?", "파이썬은 읽기 쉬운 범용 프로그래밍 언어입니다."],
    "rejected": ["뭐", "그딴 거 왜 물어봐"],
}
ds = Dataset.from_dict(pref)

# DPOConfig: TRL v1.0 안정 API
cfg = DPOConfig(
    output_dir="dpo_out",
    num_train_epochs=3,
    per_device_train_batch_size=1,
    learning_rate=5e-5,
    beta=0.1,
    max_length=128,
    report_to="none",
    use_cpu=(DEVICE == "cpu"),
)

trainer = DPOTrainer(model=model, args=cfg, train_dataset=ds, processing_class=tok)
print("\n── DPO 선호 정렬 학습 시작 ──")
trainer.train()
print("\n✅ DPO 완료: 모델이 chosen 응답을 선호하도록 정렬됨")

print("""
ℹ️ PPO(전통 RLHF)를 꼭 써야 한다면 (TRL v1.0 기준):
     from trl.experimental.ppo import PPOTrainer, PPOConfig
   단, 보상모델 학습 + 롤아웃이 필요하고 experimental 트랙이라 강의엔 DPO 권장.
🎯 결론: 선호 정렬(DPO)을 Windows(cuda/cpu)·macOS(mps) 동일 코드로 실행""")
