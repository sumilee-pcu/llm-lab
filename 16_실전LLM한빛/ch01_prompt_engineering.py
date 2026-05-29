"""
실전 LLM (한빛, 2판) — 프롬프트 엔지니어링
강의 요소: 프롬프트 엔지니어링 (zero-shot / few-shot / CoT)

【교재→2026 업데이트】
  - ChatOpenAI → ChatGoogleGenerativeAI (gemini-2.5-flash)
  - 기법별 프롬프트를 LCEL 체인으로 비교
"""
import os
from dotenv import load_dotenv
load_dotenv()

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

llm = ChatGoogleGenerativeAI(model=os.getenv("GEMINI_MODEL", "gemini-2.5-flash"), temperature=0)
parse = StrOutputParser()

# 1. Zero-shot
print("=== 1. Zero-shot ===")
zs = ChatPromptTemplate.from_template("다음 리뷰의 감정을 긍정/부정 중 하나로만: {review}") | llm | parse
print(zs.invoke({"review": "배송도 빠르고 품질도 최고예요."}).strip()[:20])

# 2. Few-shot
print("\n=== 2. Few-shot ===")
fs = ChatPromptTemplate.from_messages([
    ("system", "리뷰 감정을 긍정/부정으로 분류."),
    ("human", "별로예요"), ("ai", "부정"),
    ("human", "최고예요"), ("ai", "긍정"),
    ("human", "{review}"),
]) | llm | parse
print(fs.invoke({"review": "그냥 그래요, 다시는 안 살 듯"}).strip()[:20])

# 3. Chain-of-Thought
print("\n=== 3. CoT (단계적 추론) ===")
cot = ChatPromptTemplate.from_template(
    "단계적으로 생각한 뒤 마지막 줄에 '정답: N'만 쓰세요.\n"
    "문제: 사과 3개를 2명이 똑같이 나누면 1명당 몇 개? 소수로."
) | llm | parse
print(cot.invoke({}).strip()[-60:])

print("\n✅ ch01 프롬프트 엔지니어링 실습 완료")
