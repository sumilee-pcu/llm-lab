"""
GPU 구간 — 지식 증류 (Knowledge Distillation)   [Windows · macOS 듀얼 백엔드]
해당 교재: 23 트랜스포머를 활용한 자연어 처리 (파인튜닝·증류)

【교재 → 2026 업데이트 · Windows 경로 팩트체크】
  - teacher(큰 모델) logits → student(작은 모델) 로 KL 증류
  - transformers v5(2026-01) Trainer 기준:
      · `compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None)`
        → v5는 `num_items_in_batch` 인자를 넘기므로 시그니처에 포함해야 TypeError 없음
      · `eval_strategy`(구 evaluation_strategy), `report_to="none"`, `use_cpu`
  - 디바이스 자동: Windows/Colab=cuda→cpu, macOS=mps

실행:  python 04_distillation.py
"""
import os
from backend import print_env, torch_device

print_env()

import torch
import torch.nn.functional as F
from datasets import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

DEVICE = torch_device()
print(f"학습 디바이스: {DEVICE}")

TEACHER_ID = "distilbert-base-uncased"   # 데모용. 실제론 파인튜닝된 큰 모델을 teacher로.
STUDENT_ID = "distilbert-base-uncased"   # 데모 단순화를 위해 동일 아키텍처 사용
TEMPERATURE = 2.0
ALPHA = 0.5  # CE(정답) vs KL(teacher) 가중치

texts = ["great product love it", "fast and reliable", "highly recommend",
         "terrible quality", "waste of money", "very disappointing"]
labels = [1, 1, 1, 0, 0, 0]

tok = AutoTokenizer.from_pretrained(STUDENT_ID)
teacher = AutoModelForSequenceClassification.from_pretrained(TEACHER_ID, num_labels=2).to(DEVICE).eval()
student = AutoModelForSequenceClassification.from_pretrained(STUDENT_ID, num_labels=2).to(DEVICE)

ds = Dataset.from_dict({"text": texts, "label": labels}).map(
    lambda b: tok(b["text"], truncation=True, padding="max_length", max_length=24), batched=True)


class DistillTrainer(Trainer):
    # ← v5: num_items_in_batch 인자 필수 (없으면 TypeError)
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs["labels"]
        out = model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"])
        with torch.no_grad():
            t_logits = teacher(input_ids=inputs["input_ids"],
                               attention_mask=inputs["attention_mask"]).logits
        ce = F.cross_entropy(out.logits, labels)
        kl = F.kl_div(
            F.log_softmax(out.logits / TEMPERATURE, dim=-1),
            F.softmax(t_logits / TEMPERATURE, dim=-1),
            reduction="batchmean",
        ) * (TEMPERATURE ** 2)
        loss = ALPHA * ce + (1 - ALPHA) * kl
        return (loss, out) if return_outputs else loss


args = TrainingArguments(
    output_dir=os.path.join(os.path.dirname(__file__), "distill_out"),
    num_train_epochs=5,
    per_device_train_batch_size=3,
    learning_rate=5e-5,
    logging_steps=2,
    report_to="none",
    use_cpu=(DEVICE == "cpu"),
)

trainer = DistillTrainer(model=student, args=args, train_dataset=ds)
print("\n── 증류 학습 시작 (teacher logits → student) ──")
trainer.train()
print("\n✅ 증류 완료: student가 teacher 분포를 모방하도록 학습됨")
print("🎯 결론: KL 기반 지식 증류를 Windows(cuda/cpu)·macOS(mps) 동일 코드로 실행")
