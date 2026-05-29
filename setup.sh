#!/bin/bash
# ============================================================
# LLM 실습 환경 세팅 스크립트
# 대상: MacBook / Python 3.11+
# 사용법: bash setup.sh
# ============================================================

set -e

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo " LLM 실습 환경 구성 시작"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# 1. Python 버전 확인
echo "\n[1/5] Python 버전 확인..."
python3 --version || { echo "❌ Python3 가 설치되어 있지 않습니다. https://python.org 에서 설치하세요."; exit 1; }

# 2. 가상환경 생성
echo "\n[2/5] 가상환경 생성 (.venv)..."
if [ ! -d ".venv" ]; then
    python3 -m venv .venv
    echo "✅ 가상환경 생성 완료"
else
    echo "ℹ️  기존 가상환경 재사용"
fi

# 3. 가상환경 활성화 + pip 업그레이드
echo "\n[3/5] 가상환경 활성화 & pip 업그레이드..."
source .venv/bin/activate
pip install --upgrade pip -q

# 4. 패키지 설치
echo "\n[4/5] 패키지 설치 중 (requirements.txt)..."
pip install -r requirements.txt

# 5. Jupyter 커널 등록
echo "\n[5/5] Jupyter 커널 등록..."
python -m ipykernel install --user --name=llm-practice --display-name "LLM실습 (Python 3)"

echo "\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo " ✅ 환경 구성 완료!"
echo " 실행: source .venv/bin/activate"
echo "       jupyter lab"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
