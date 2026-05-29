"""
두잇! LLM 활용 에이전트 입문 — 3장 실습
: LangChain 체인·프롬프트 템플릿

【교재→2026 업데이트】
  - LangChain 0.x → 1.x (2025-10 릴리즈)
  - LLMChain 제거됨 → LCEL (| 파이프라인) 사용
    교재: chain = LLMChain(llm=llm, prompt=prompt)
    현재: chain = prompt | llm | StrOutputParser()
  - from langchain.prompts → from langchain_core.prompts
"""
import os
from dotenv import load_dotenv
load_dotenv()

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda

llm = ChatGoogleGenerativeAI(model=os.getenv("GEMINI_MODEL", "gemini-2.5-flash"), temperature=0.3)

# 1. LCEL 기본 체인
prompt = ChatPromptTemplate.from_messages([
    ("system", "당신은 {role} 전문가입니다. 간결하게 답변해주세요."),
    ("human", "{question}"),
])
chain = prompt | llm | StrOutputParser()
result = chain.invoke({"role": "파이썬", "question": "리스트 컴프리헨션이란?"})
print(result)

# 2. 체인 연결 (요약 → 번역)
summary_prompt = ChatPromptTemplate.from_template("{text}를 한 줄로 요약해줘.")
translate_prompt = ChatPromptTemplate.from_template("{text}를 영어로 번역해줘.")
summary_chain = summary_prompt | llm | StrOutputParser()
translate_chain = translate_prompt | llm | StrOutputParser()
pipeline = summary_chain | RunnableLambda(lambda x: {"text": x}) | translate_chain
result = pipeline.invoke({"text": "LangChain은 LLM 기반 애플리케이션을 만들기 위한 프레임워크입니다."})
print(result)

# 3. 배치 처리
questions = [
    {"role": "역사", "question": "조선 건국 연도는?"},
    {"role": "수학", "question": "피타고라스 정리란?"},
]
results = chain.batch(questions)
for q, r in zip(questions, results):
    print(f"[{q['role']}] {r.strip()[:60]}")

print("\n✅ ch03 LangChain 체인·프롬프트 템플릿 실습 완료")
