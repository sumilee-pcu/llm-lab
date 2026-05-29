"""
LangChain을 활용한 Google Cloud의 생성 AI — LangChain 오케스트레이션 기초
강의 요소: LangChain 오케스트레이션 / 기초 모델 선택, 프롬프트 디자인

【교재→2026 업데이트】
  - Vertex AI (PaLM 2 / ChatVertexAI) → ChatGoogleGenerativeAI(Gemini)
    : GCP 프로젝트·Vertex API 활성화 없이 GOOGLE_API_KEY 만으로 실행 (로컬/무료 친화)
  - 고급 프롬프트 디자인을 LCEL 체인으로 구성
"""
import os
from dotenv import load_dotenv
load_dotenv()

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda

llm = ChatGoogleGenerativeAI(model=os.getenv("GEMINI_MODEL", "gemini-2.5-flash"), temperature=0)

# 1. 신속 프롬프트 엔지니어링 (few-shot)
print("=== 1. Few-shot 프롬프트 ===")
fewshot = ChatPromptTemplate.from_messages([
    ("system", "회사명을 받으면 한 줄 소개를 생성합니다."),
    ("human", "Google"), ("ai", "Google: 검색·클라우드·AI를 선도하는 글로벌 기술 기업."),
    ("human", "{company}"),
])
chain = fewshot | llm | StrOutputParser()
print(chain.invoke({"company": "한빛미디어"}).strip()[:80])

# 2. 오케스트레이션: 분석 → 요약 2단계 파이프
print("\n=== 2. 2단계 오케스트레이션 (분석→요약) ===")
analyze = ChatPromptTemplate.from_template("{text}의 핵심 키워드 3개를 쉼표로.") | llm | StrOutputParser()
summarize = ChatPromptTemplate.from_template("키워드 [{kw}]로 한 문장 요약을 만들어줘.") | llm | StrOutputParser()
pipe = analyze | RunnableLambda(lambda kw: {"kw": kw}) | summarize
out = pipe.invoke({"text": "생성형 AI는 클라우드 인프라 위에서 RAG와 에이전트로 확장된다."})
print(out.strip()[:120])

print("\n✅ ch01 LangChain 오케스트레이션 실습 완료")
