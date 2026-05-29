"""
테디노트 RAG 비법노트 — 심화편 — PART03 RAG 평가와 개선
강의 요소: RAG 평가·개선 체계(품질 수치화)

【교재→2026 업데이트】
  - RAGAS 류 평가를 LLM-as-judge로 경량 구현 (context_relevance, faithfulness)
  - 모든 호출 Gemini
"""
import os, re
from dotenv import load_dotenv
load_dotenv()

from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

llm = ChatGoogleGenerativeAI(model=os.getenv("GEMINI_MODEL", "gemini-2.5-flash"), temperature=0)
embeddings = GoogleGenerativeAIEmbeddings(model=os.getenv("GEMINI_EMBEDDING_MODEL", "models/gemini-embedding-2"))

DOCS = ["RAG 평가는 검색 적합도와 답변 충실도를 함께 본다.",
        "충실도(faithfulness)는 답변이 컨텍스트에 근거하는 정도다.",
        "검색 적합도는 가져온 문서가 질문과 관련된 정도다."]
retriever = Chroma.from_documents([Document(page_content=t) for t in DOCS],
                                  embedding=embeddings).as_retriever(search_kwargs={"k": 2})

answer_chain = ChatPromptTemplate.from_template(
    "컨텍스트로만 답하세요.\n[컨텍스트]\n{ctx}\n[질문] {q}") | llm | StrOutputParser()
judge = ChatPromptTemplate.from_template(
    "질문과 컨텍스트, 답변을 보고 {metric}를 1~5점으로 평가. 'SCORE: N'만 끝에.\n"
    "[질문]{q}\n[컨텍스트]{ctx}\n[답변]{a}") | llm | StrOutputParser()


def score_of(text):
    m = re.search(r"SCORE:\s*([1-5])", text)
    return int(m.group(1)) if m else None


q = "faithfulness가 뭐야?"
ctx = "\n".join(d.page_content for d in retriever.invoke(q))
a = answer_chain.invoke({"ctx": ctx, "q": q}).strip()
print(f"답변: {a[:100]}\n")
for metric in ["검색 적합도(context relevance)", "답변 충실도(faithfulness)"]:
    s = score_of(judge.invoke({"metric": metric, "q": q, "ctx": ctx, "a": a}))
    print(f"  {metric}: {s} / 5")

print("\n✅ ch02 RAG 평가 실습 완료")
