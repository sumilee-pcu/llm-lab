"""
미니 벤치마크 하네스 — 논문 데이터 PoC
=====================================
같은 모델·데이터·코드로 '디바이스만' 바꿔 N회 반복 측정 → 평균±표준편차 표 생성.
데모(1회 실행)와 논문 데이터(통제된 반복실험 + 비교표)의 차이를 실물로 보여주기 위함.

측정 지표 (논문 표에 들어갈 것들):
  - train_sec      : 학습 wall-clock(초)
  - throughput     : samples/sec
  - peak_mem_mb    : 피크 메모리(MB) — mps는 torch API, 그 외는 측정 생략(-)
  - eval_acc       : 최종 평가 정확도

사용:  python bench_classification.py --devices mps,cpu --repeats 3
출력:  results_bench.tsv  (+ 콘솔 요약표)
"""
import argparse
import gc
import os
import statistics
import time

import numpy as np
import torch
from datasets import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

MODEL_ID = "distilbert-base-uncased"
SEED = 42

SAMPLES = [
    ("fast delivery and great quality", 1), ("absolutely love it", 1),
    ("best purchase ever", 1), ("highly recommend this", 1),
    ("works perfectly i am happy", 1), ("excellent product", 1),
    ("terrible never buying again", 0), ("very disappointing quality", 0),
    ("worst experience ever", 0), ("i want a refund", 0),
    ("broke after one day", 0), ("complete waste of money", 0),
]


def run_once(device: str, seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_ID, num_labels=2).to(device)

    ds = Dataset.from_dict({"text": [t for t, _ in SAMPLES], "label": [y for _, y in SAMPLES]})
    ds = ds.map(lambda b: tok(b["text"], truncation=True, padding="max_length", max_length=32), batched=True)
    ds = ds.train_test_split(test_size=0.25, seed=seed)

    def metrics(p):
        return {"accuracy": float((np.argmax(p.predictions, axis=1) == p.label_ids).mean())}

    args = TrainingArguments(
        output_dir=os.path.join(os.path.dirname(__file__), "_bench_tmp"),
        eval_strategy="no", num_train_epochs=8,
        per_device_train_batch_size=4, learning_rate=5e-4,
        logging_strategy="no", report_to="none", use_cpu=(device == "cpu"),
        save_strategy="no",
    )
    trainer = Trainer(model=model, args=args, train_dataset=ds["train"],
                      eval_dataset=ds["test"], compute_metrics=metrics)

    if device == "mps":
        torch.mps.empty_cache()

    t0 = time.perf_counter()
    trainer.train()
    train_sec = time.perf_counter() - t0

    acc = trainer.evaluate()["eval_accuracy"]
    n_samples = len(ds["train"]) * 8  # epochs
    throughput = n_samples / train_sec
    # mps 메모리: 학습 직후 드라이버 할당량(MB)을 근사 피크로 사용
    peak_mem = torch.mps.driver_allocated_memory() / 1e6 if device == "mps" else None

    del model, trainer
    gc.collect()
    if device == "mps":
        torch.mps.empty_cache()
    return {"train_sec": train_sec, "throughput": throughput,
            "peak_mem_mb": peak_mem, "eval_acc": acc}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--devices", default="mps,cpu")
    ap.add_argument("--repeats", type=int, default=3)
    a = ap.parse_args()

    avail = {"cpu": True, "mps": torch.backends.mps.is_available(),
             "cuda": torch.cuda.is_available()}
    devices = [d for d in a.devices.split(",") if avail.get(d)]
    print(f"측정 디바이스: {devices} (repeats={a.repeats}, model={MODEL_ID})\n")

    rows = []
    for dev in devices:
        runs = [run_once(dev, SEED + i) for i in range(a.repeats)]
        agg = {}
        for k in ["train_sec", "throughput", "eval_acc"]:
            vals = [r[k] for r in runs]
            agg[k] = (statistics.mean(vals), statistics.pstdev(vals))
        mems = [r["peak_mem_mb"] for r in runs if r["peak_mem_mb"] is not None]
        agg["peak_mem_mb"] = (statistics.mean(mems), 0) if mems else (None, None)
        rows.append((dev, agg))

    # 콘솔 표
    print(f"{'device':<6} {'train_sec':>16} {'samples/sec':>16} {'peak_mem_MB':>14} {'eval_acc':>14}")
    print("-" * 70)
    for dev, agg in rows:
        ts = f"{agg['train_sec'][0]:.2f}±{agg['train_sec'][1]:.2f}"
        tp = f"{agg['throughput'][0]:.1f}±{agg['throughput'][1]:.1f}"
        mm = f"{agg['peak_mem_mb'][0]:.0f}" if agg['peak_mem_mb'][0] else "-"
        ac = f"{agg['eval_acc'][0]:.2f}±{agg['eval_acc'][1]:.2f}"
        print(f"{dev:<6} {ts:>16} {tp:>16} {mm:>14} {ac:>14}")

    # TSV 저장 (논문 표 원본)
    out = os.path.join(os.path.dirname(__file__), "results_bench.tsv")
    with open(out, "w") as f:
        f.write("device\trepeats\ttrain_sec_mean\ttrain_sec_std\tthroughput_mean\tthroughput_std\tpeak_mem_mb\teval_acc_mean\teval_acc_std\n")
        for dev, agg in rows:
            mm = f"{agg['peak_mem_mb'][0]:.1f}" if agg['peak_mem_mb'][0] else ""
            f.write(f"{dev}\t{a.repeats}\t{agg['train_sec'][0]:.3f}\t{agg['train_sec'][1]:.3f}\t"
                    f"{agg['throughput'][0]:.2f}\t{agg['throughput'][1]:.2f}\t{mm}\t"
                    f"{agg['eval_acc'][0]:.3f}\t{agg['eval_acc'][1]:.3f}\n")
    print(f"\n📄 결과 저장: {out}")


if __name__ == "__main__":
    main()
