"""
GPU 구간 — 스테이블 디퓨전 이미지 생성   [Windows · macOS 듀얼 백엔드]
해당 교재: 24 파이토치로 배우는 LLM & AI (스테이블 디퓨전)

【교재 → 2026 업데이트 · Windows 경로 팩트체크】
  - diffusers(2026, 0.33+)는 한 코드로 cuda/mps/cpu 모두 지원 → 별도 MLX 분기 불필요
      · Windows/Colab : cuda
      · macOS         : mps (Apple Silicon GPU에서 네이티브 생성)
      · CPU           : 매우 느림(데모만)
  - 빠른 데모를 위해 'sd-turbo'(1~4 step) 사용. 교재의 SD1.5/SDXL은 step↑.
  - MLX 네이티브 대안(선택): FLUX 계열은 'mflux'(ml-explore) 로 Apple Silicon 가속 가능.

실행:  python 05_diffusion.py
설치:  pip install diffusers transformers accelerate torch
주의:  최초 1회 모델 다운로드(수 GB) 발생.
"""
from backend import print_env, torch_device

print_env()

import torch
from diffusers import AutoPipelineForText2Image

DEVICE = torch_device()
print(f"생성 디바이스: {DEVICE}")

# sd-turbo: 1~4 step 고속 생성. mps/cuda는 float16, cpu는 float32.
dtype = torch.float16 if DEVICE in ("cuda", "mps") else torch.float32
pipe = AutoPipelineForText2Image.from_pretrained(
    "stabilityai/sd-turbo", torch_dtype=dtype
).to(DEVICE)

prompt = "a cute robot teaching a class, digital art"
print(f'\n프롬프트: "{prompt}" 생성 중...')
# sd-turbo는 guidance_scale=0.0, 적은 step 권장
image = pipe(prompt, num_inference_steps=2, guidance_scale=0.0).images[0]
out_path = "diffusion_out.png"
image.save(out_path)
print(f"✅ 이미지 저장: {out_path} ({image.size[0]}x{image.size[1]})")
print("🎯 결론: 텍스트→이미지 생성을 Windows(cuda)·macOS(mps) 동일 코드로 실행")
