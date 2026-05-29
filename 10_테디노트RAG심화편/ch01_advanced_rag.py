"""
테디노트 RAG 비법노트 — 심화편 — PART01/02 고급 RAG & LCEL 고급
강의 요소: 고급 RAG(멀티쿼리·하이브리드) / LCEL 고급 문법

【교재→2026 업데이트】
  - OpenAIEmbeddings → GoogleGenerativeAIEmbeddings (models/gemini-embedding-2)
  - 멀티쿼리: 질문을 여러 변형으로 확장해 검색 재현율↑ (LCEL로 구성)
"""
import os
from dotenv import load_dotenv
load_dotenv()

from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

llm = ChatGoogleGenerativeAI(model=os.getenv("GEMINI_MODEL", "gemini-2.5-flash"), temperature=0)
embeddings = GoogleGenerativeAIEmbeddings(model=os.getenv("GEMINI_EMBEDDING_MODEL", "models/gemini-embedding-2"))

DOCS = [
    "하이브리드 검색은 키워드(BM25)와 임베딩(시맨틱) 검색을 결합합니다.",
    "멀티쿼리는 하나의 질문을 여러 표현으로 확장해 검색 누락을 줄입니다.",
    "리랭커는 1차 검색 결과를 질문 적합도 순으로 재정렬합니다.",
    "청크 크기와 오버랩은 검색 품질에 큰 영향을 줍니다.",
]
vectorstore = Chroma.from_documents([Document(page_content=t) for t in DOCS], embedding=embeddings)

# 멀티쿼리 생성 체인
mq_chain = ChatPromptTemplate.from_template(
    "다음 질문을 의미가 같은 서로 다른 표현 3개로 바꿔 줄바꿈으로만 출력:\n{q}"
) | llm | StrOutputParser()


def multiquery_search(question: str, k: int = 2):
    variants = [question] + [v.strip() for v in mq_chain.invoke({"q": question}).split("\n") if v.strip()]
    seen, results = set(), []
    for v in variants:
        for d in vectorstore.similarity_search(v, k=k):
            if d.page_content not in seen:
                seen.add(d.page_content); results.append(d.page_content)
    return variants, results


variants, docs = multiquery_search("검색 누락을 줄이는 방법은?")
print("생성된 쿼리 변형:")
for v in variants:
    print("  -", v[:50])
print(f"\n수집된 고유 문서 {len(docs)}개:")
for d in docs:
    print("  →", d)

print("\n✅ ch01 고급 RAG(멀티쿼리) 실습 완료")
