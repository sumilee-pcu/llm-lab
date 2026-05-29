"""
실전 RAG 기반 생성형 AI 개발 — 모듈형 RAG
강의 요소: 모듈형 RAG (질문 재작성 → 검색 → 재순위 → 생성)

【교재→2026 업데이트】
  - 단일 RetrievalQA → 모듈 단계를 명시적으로 분리한 LCEL 파이프
  - 각 모듈(rewrite/retrieve/rerank/generate)을 교체 가능하게 구성
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

DOCS = ["모듈형 RAG는 검색·생성 단계를 독립 모듈로 분리한다.",
        "질문 재작성은 모호한 질의를 검색 친화적으로 바꾼다.",
        "재순위(rerank)는 검색 결과를 적합도 순으로 다시 정렬한다.",
        "생성 모듈은 최종 컨텍스트로 답을 만든다."]
vectorstore = Chroma.from_documents([Document(page_content=t) for t in DOCS], embedding=embeddings)

rewrite = ChatPromptTemplate.from_template("검색에 좋게 한 줄로 다시 써줘: {q}") | llm | StrOutputParser()
rerank = ChatPromptTemplate.from_template(
    "질문: {q}\n문서들:\n{docs}\n가장 관련 높은 문서 1개의 내용만 그대로 출력.") | llm | StrOutputParser()
generate = ChatPromptTemplate.from_template("컨텍스트로만 답: {ctx}\n질문: {q}") | llm | StrOutputParser()


def modular_rag(question: str):
    rq = rewrite.invoke({"q": question}).strip()
    cand = vectorstore.similarity_search(rq, k=3)
    docs_text = "\n".join(f"- {d.page_content}" for d in cand)
    top = rerank.invoke({"q": question, "docs": docs_text}).strip()
    ans = generate.invoke({"ctx": top, "q": question}).strip()
    return rq, top, ans


rq, top, ans = modular_rag("재정렬이 왜 필요해?")
print(f"① 재작성: {rq[:50]}")
print(f"② 재순위 선택: {top[:50]}")
print(f"③ 생성 답변: {ans[:100]}")

print("\n✅ ch02 모듈형 RAG 실습 완료")
