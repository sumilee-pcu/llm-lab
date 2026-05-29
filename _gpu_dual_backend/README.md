# GPU 구간 듀얼 백엔드 블루프린트 (Windows · macOS 공용)

Tier 2 교재(25·24·23·09·27·06·15)의 **GPU 의존 구간**(파인튜닝·증류·디퓨전·양자화·RLHF)을
**개념 주석으로 우회하지 않고 실제로 실행**하기 위한 공용 패턴입니다.

## 핵심 원칙

> 하나의 레슨 · 같은 데이터 · 같은 출력 지표 → **백엔드/디바이스만 자동 전환**

| 환경 | 경로 | 비고 |
|------|------|------|
| Apple Silicon (M-시리즈) | **MLX**(`mlx-lm`) 또는 **torch(mps)** | 칩 내장 GPU 네이티브, NVIDIA 불필요 |
| Windows / Intel / Colab | **torch**(`transformers`/`peft`/`trl`/`diffusers`) | CUDA 있으면 GPU, 없으면 CPU 소형 모델 |

`backend.py`가 환경을 자동 감지 → 수강생은 **분기 코드를 만지지 않고** 자기 OS에서 그대로 실행.
PyTorch 자체가 Apple Silicon의 `mps`를 지원하므로, MLX turnkey가 없는 작업(분류·증류·디퓨전·DPO)은
**동일 torch 코드가 Mac은 mps, Windows는 cuda/cpu** 로 돈다(= "두 가지 경우").

## 파일 → 교재 매핑

| 파일 | GPU 주제 | 해당 교재 | macOS | Windows(팩트체크) | 검증 |
|------|---------|----------|-------|------------------|------|
| `01_finetune_lora.py` | LoRA/PEFT 생성 파인튜닝 | 27·06·15·24 | MLX(`mlx-lm`) | `transformers`+`peft` | ✅ 양쪽 실행 |
| `02_quantize.py` | 양자화(추론 최적화) | 06·15 | MLX 4bit | CUDA bnb / CPU 동적양자화 | ✅ MLX 실행 |
| `03_classification_finetune.py` | 텍스트 분류 파인튜닝 | 25·23·09 | torch(mps) | `transformers` v5 Trainer | ✅ 실행 |
| `04_distillation.py` | 지식 증류 | 23 | torch(mps) | v5 custom `compute_loss` | ✅ 실행 |
| `05_diffusion.py` | 스테이블 디퓨전 | 24 | torch(mps) | `diffusers` cuda | API 검증(모델 다운로드 시 실행) |
| `06_rlhf_dpo.py` | RLHF / 선호정렬(DPO) | 27 | torch(mps) | `trl` v1.0 DPO | ✅ 실행 |

> 09·15는 RAG/추론최적화 등 비-GPU 구간이 많아, GPU 부분만 위 파일을 공유 재사용.

## 🔎 Windows 경로 최신 팩트체크 (2026-05 기준, 지식컷오프 이후 변경 반영)

강의 코드가 **구버전 API로 깨지지 않도록** 최신 라이브러리 변경을 직접 확인해 반영했습니다.

- **transformers v5.0** (2026-01-26, 5년 만의 메이저)
  - `evaluation_strategy` **삭제 → `eval_strategy`**
  - `no_cuda` 삭제 → `use_cpu`, `per_gpu_*` 삭제 → `per_device_*`
  - 로깅 자동감지 폐지 → **`report_to="none"` 명시 필수**
  - TF/Flax 전면 제거(PyTorch 전용), **Python 3.10+/PyTorch 2.4+**
  - Trainer `compute_loss`에 **`num_items_in_batch` 인자 추가** (증류 등 커스텀 손실 시 시그니처 필수)
  - 주간 릴리스마다 breaking → **버전 핀 고정 권장**
- **TRL v1.0** (2026-03)
  - 안정: SFT/**DPO**/Reward/RLOO/GRPO
  - **PPO 이동**: `from trl.experimental.ppo import PPOTrainer, PPOConfig` (experimental)
  - 강의 표준은 PPO 대신 **DPO** (보상모델·롤아웃 불필요)
- **PEFT 0.18+** — transformers v5 호환 최소 버전
- **diffusers 0.33+** (검증 환경 0.37.1) — cuda/mps/cpu 단일 코드, `AutoPipelineForText2Image`
- **bitsandbytes** — 멀티백엔드(CPU/Windows) 아직 **alpha** → 4bit는 **Colab(CUDA)** 권장, Windows CPU는 동적양자화 fallback

## 사용법

```bash
pip install -r requirements-gpu.txt   # 환경별 자동 분기 설치
python backend.py                      # 어떤 백엔드/디바이스가 잡히는지 확인
python 03_classification_finetune.py   # 어느 OS든 동일 명령
```

## ✅ 검증 결과 (2026-05-29, Apple M5 Max / 128GB)

| 파일 | 결과 |
|------|------|
| 01 LoRA(MLX) | val loss 4.509 → 0.831, 어댑터 저장 |
| 01 LoRA(torch) | tiny-gpt2, mps/cpu 학습 step 정상 |
| 02 양자화(MLX) | 4.5 bits/weight, 276MB |
| 03 분류(torch v5) | mps 학습, 정확도 0.67, "this is really good"→긍정 |
| 04 증류(torch v5) | KL 손실 train_loss 0.31 |
| 06 DPO(trl 1.5.1) | rewards/chosen 1.005 > rejected −0.27, accuracies 1.0 |
| trl import | DPO/SFT 안정 + PPO experimental 경로 확인 |
| diffusers import | 0.37.1, AutoPipelineForText2Image 확인 |

## 강의 운영 팁

- **강사 라이브 데모**: M-시리즈 맥 → MLX/mps로 빠른 실제 학습 시연.
- **Windows 수강생**: 같은 파일 그대로 → CPU 소형 모델로 개념 체득, "진짜 규모는 Colab" 병행.
- **Colab 수강생**: torch 경로 + `torch.cuda` 자동 사용 → 더 큰 모델로 확장.
- **풀 PPO RLHF** 등 진짜 대규모만 필요한 구간은 DPO로 대체하거나 개념 주석 유지.
