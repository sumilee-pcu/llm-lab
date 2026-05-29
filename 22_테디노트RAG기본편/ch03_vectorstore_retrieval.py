"""
테디노트 RAG 비법노트 — 기본편 — 3. 벡터 저장·검색
(원본 저장소: github.com/teddylee777/langchain-kr — CH07 벡터스토어 / CH08 리트리버 대응)

【교재→2026 업데이트】
  - FAISS / Chroma(persist) → Chroma 인메모리 (Docker/디스크 의존성 제거)
  - OpenAIEmbeddings → GoogleGenerativeAIEmbeddings (models/gemini-embedding-2)
  - as_retriever(search_kwargs={"k": ...}) — 1.x 동일
"""
import os
from dotenv import load_dotenv
load_dotenv()

from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document

EMBEDDING_MODEL = os.getenv("GEMINI_EMBEDDING_MODEL", "models/gemini-embedding-2")
embeddings = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL)

# 1. 지식 문서 (인라인)
knowledge = [
    "RAG는 검색 증강 생성으로, 외부 문서를 검색해 LLM 답변 근거로 사용합니다.",
    "임베딩은 텍스트를 고차원 벡터로 변환하며, 의미가 비슷하면 벡터 거리도 가깝습니다.",
    "벡터 스토어는 임베딩을 저장하고 유사도 기반으로 검색하는 데이터베이스입니다.",
    "리트리버는 질의와 유사한 문서를 k개 반환하는 검색 컴포넌트입니다.",
    "LCEL은 prompt | llm | parser 형태로 RAG 체인을 간결하게 구성합니다.",
]
docs = [Document(page_content=t, metadata={"id": i}) for i, t in enumerate(knowledge)]

# 2. 벡터 스토어 생성
print("=== 1. Chroma 벡터 스토어 생성 ===")
vectorstore = Chroma.from_documents(documents=docs, embedding=embeddings)
print(f"저장된 문서 수: {len(docs)}")

# 3. 유사도 검색
print("\n=== 2. 유사도 검색 (k=2) ===")
retriever = vectorstore.as_retriever(search_kwargs={"k": 2})
query = "리트리버가 뭐하는 거야?"
results = retriever.invoke(query)
for r in results:
    print(f"  → {r.page_content}")

# 4. score 포함 검색
print("\n=== 3. 점수 포함 검색 ===")
scored = vectorstore.similarity_search_with_score("임베딩과 벡터 거리", k=2)
for doc, score in scored:
    print(f"  ({score:.3f}) {doc.page_content[:40]}")

print("\n✅ ch03 벡터 저장·검색 실습 완료")
