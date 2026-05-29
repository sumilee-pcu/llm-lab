"""
LangChain을 활용한 Google Cloud의 생성 AI — RAG·그라운딩(환각 완화)
강의 요소: 임베딩·벡터 저장소 / 환각·그라운딩 응답

【교재→2026 업데이트】
  - Vertex AI 벡터 검색 / pgVector → Chroma 인메모리 + Gemini Embeddings
  - 그라운딩: 컨텍스트에 없으면 '모른다'고 답하도록 강제(환각 방지)
"""
import os
from dotenv import load_dotenv
load_dotenv()

from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

llm = ChatGoogleGenerativeAI(model=os.getenv("GEMINI_MODEL", "gemini-2.5-flash"), temperature=0)
embeddings = GoogleGenerativeAIEmbeddings(model=os.getenv("GEMINI_EMBEDDING_MODEL", "models/gemini-embedding-2"))

DOCS = [
    "Gemini는 구글의 멀티모달 LLM이며 텍스트·이미지·코드를 처리합니다.",
    "Vertex AI는 구글 클라우드의 ML 플랫폼으로 모델 배포·서빙을 지원합니다.",
    "그라운딩은 모델 응답을 신뢰 가능한 데이터에 근거시키는 기법입니다.",
]
retriever = Chroma.from_documents([Document(page_content=t) for t in DOCS],
                                  embedding=embeddings).as_retriever(search_kwargs={"k": 2})

# 그라운딩 프롬프트: 컨텍스트 밖이면 모른다고 답하게 함
prompt = ChatPromptTemplate.from_template(
    "아래 컨텍스트에 근거해서만 답하세요. 컨텍스트에 없으면 정확히 '제공된 자료에 없습니다'라고 답하세요.\n"
    "[컨텍스트]\n{context}\n\n[질문] {question}")
rag = ({"context": retriever | (lambda ds: "\n".join(d.page_content for d in ds)),
        "question": RunnablePassthrough()} | prompt | llm | StrOutputParser())

print("=== 그라운딩 RAG (환각 완화) ===")
print("Q1(자료 내):", rag.invoke("그라운딩이 뭐야?").strip()[:100])
print("\nQ2(자료 밖):", rag.invoke("애플의 시가총액은 얼마야?").strip()[:100])  # → '제공된 자료에 없습니다'

print("\n✅ ch02 RAG·그라운딩 실습 완료")
