"""
GPU 구간 — 텍스트 분류 파인튜닝   [Windows · macOS 듀얼 백엔드]
해당 교재: 25 핸즈온 LLM(파인튜닝 입문) · 23 트랜스포머 NLP(분류·NER) · 09 NLP와 LLM(ML 분류)

【교재 → 2026 업데이트 · Windows 경로 팩트체크】
  교재는 HF Trainer 기반 (Colab GPU 전제). 2026-05 기준 최신 API로 갱신:
    - transformers v5.0(2026-01): `evaluation_strategy` 삭제 → **`eval_strategy`**
    - 로깅 자동감지 폐지 → **`report_to="none"` 명시 필수**
    - `per_gpu_*` 삭제 → `per_device_*` 사용, PyTorch 전용(2.4+)
  실행 디바이스 자동 선택:
    - Windows/Colab : cuda(있으면) → cpu
    - macOS         : mps(Apple Silicon GPU) — PyTorch가 칩 GPU를 직접 사용
  ※ 분류 파인튜닝은 MLX에 turnkey 트레이너가 없어 양 플랫폼 모두 torch 경로를 쓰되,
     Mac은 mps로 자동 가속됩니다(동일 코드, device만 다름 = '두 가지 경우').

실행:  python 03_classification_finetune.py
설치:  pip install -r requirements-gpu.txt
"""
import os
from backend import print_env, torch_device

print_env()

import numpy as np
import torch
from datasets import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

DEVICE = torch_device()
print(f"학습 디바이스: {DEVICE}")

# 강의용 소형 모델 — fast 토크나이저 내장(transformers v5 호환). CPU에서도 수 분 내.
# 더 가볍게: 'prajjwal1/bert-tiny'는 fast 토크나이저 파일이 없어 v5에서 변환 실패하므로 사용 금지.
MODEL_ID = "distilbert-base-uncased"

# 미니 감성분류 데이터 (0=부정, 1=긍정)
# distilbert-base-uncased 는 영어 모델 → 영어 샘플로 학습 효과를 시연.
# (한국어로 하려면 'distilbert-base-multilingual-cased' 등 다국어 모델로 교체)
samples = [
    ("fast delivery and great quality", 1), ("absolutely love it", 1),
    ("best purchase ever", 1), ("highly recommend this", 1),
    ("works perfectly i am happy", 1), ("excellent product", 1),
    ("terrible never buying again", 0), ("very disappointing quality", 0),
    ("worst experience ever", 0), ("i want a refund", 0),
    ("broke after one day", 0), ("complete waste of money", 0),
]
texts = [t for t, _ in samples]
labels = [y for _, y in samples]

tok = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_ID, num_labels=2).to(DEVICE)

ds = Dataset.from_dict({"text": texts, "label": labels})
ds = ds.map(lambda b: tok(b["text"], truncation=True, padding="max_length", max_length=32), batched=True)
ds = ds.train_test_split(test_size=0.25, seed=42)


def metrics(p):
    preds = np.argmax(p.predictions, axis=1)
    return {"accuracy": float((preds == p.label_ids).mean())}


args = TrainingArguments(
    output_dir=os.path.join(os.path.dirname(__file__), "clf_out"),
    eval_strategy="epoch",          # ← v5: evaluation_strategy 아님
    num_train_epochs=8,
    per_device_train_batch_size=4,  # ← v5: per_gpu_* 제거
    per_device_eval_batch_size=4,
    learning_rate=5e-4,
    logging_steps=2,
    report_to="none",               # ← v5: 로깅 자동감지 폐지 대응
    use_cpu=(DEVICE == "cpu"),      # ← v5: no_cuda 제거, use_cpu 사용
)

trainer = Trainer(
    model=model, args=args,
    train_dataset=ds["train"], eval_dataset=ds["test"],
    compute_metrics=metrics,
)

print("\n── 분류 파인튜닝 시작 ──")
trainer.train()
result = trainer.evaluate()
print(f"\n✅ 평가 정확도: {result['eval_accuracy']:.2f}")

# 추론 시연
model.eval()
test_text = "this is really good"
enc = tok(test_text, return_tensors="pt").to(DEVICE)
with torch.no_grad():
    pred = model(**enc).logits.argmax(-1).item()
print(f'추론: "{test_text}" → {"긍정" if pred == 1 else "부정"}')
print("\n🎯 결론: 분류 파인튜닝을 Windows(cuda/cpu)·macOS(mps) 동일 코드로 실행 완료")
