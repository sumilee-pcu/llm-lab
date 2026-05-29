"""
테디노트 RAG 비법노트 — 기본편 — 1. LangChain 기초·LCEL
(원본 저장소: github.com/teddylee777/langchain-kr — CH01~CH02 대응)

【교재→2026 업데이트】
  - ChatOpenAI / OpenAI → ChatGoogleGenerativeAI (gemini-2.5-flash)
  - LLMChain → LCEL (prompt | llm | StrOutputParser)
  - from langchain.prompts → from langchain_core.prompts
  - OPENAI_API_KEY → GOOGLE_API_KEY
"""
import os
from dotenv import load_dotenv
load_dotenv()

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser

llm = ChatGoogleGenerativeAI(model=os.getenv("GEMINI_MODEL", "gemini-2.5-flash"), temperature=0)

# 1. 가장 단순한 LCEL 체인
print("=== 1. 기본 LCEL 체인 ===")
prompt = PromptTemplate.from_template("{topic}에 대해 한 문장으로 설명해줘.")
chain = prompt | llm | StrOutputParser()
print(chain.invoke({"topic": "RAG(검색 증강 생성)"}))

# 2. 멀티 메시지 프롬프트 (system + human)
print("\n=== 2. system/human 프롬프트 ===")
chat_prompt = ChatPromptTemplate.from_messages([
    ("system", "당신은 {persona}입니다. 친절하고 간결하게 답하세요."),
    ("human", "{question}"),
])
chat_chain = chat_prompt | llm | StrOutputParser()
print(chat_chain.invoke({"persona": "RAG 전문 강사", "question": "LCEL이 왜 좋은가요?"}))

# 3. 배치 처리
print("\n=== 3. 배치(batch) ===")
topics = [{"topic": "임베딩"}, {"topic": "벡터 데이터베이스"}, {"topic": "리트리버"}]
for t, r in zip(topics, chain.batch(topics)):
    print(f"[{t['topic']}] {r.strip()[:60]}")

print("\n✅ ch01 LangChain 기초·LCEL 실습 완료")
