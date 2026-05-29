"""
실전 RAG 기반 생성형 AI 개발 — 색인·검색 최적화 (MMR vs 유사도)
강의 요소: 색인·검색 최적화

【교재→2026 업데이트】
  - 파인콘/크로마 동일 개념 → Chroma 인메모리로 검색 전략 비교
  - 유사도(similarity) vs MMR(다양성) 검색 결과 차이를 직접 확인
"""
import os
from dotenv import load_dotenv
load_dotenv()

from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document

embeddings = GoogleGenerativeAIEmbeddings(model=os.getenv("GEMINI_EMBEDDING_MODEL", "models/gemini-embedding-2"))

# 비슷한 문서가 중복적으로 많은 코퍼스 (MMR 효과 확인용)
DOCS = [
    "파이썬은 배우기 쉬운 프로그래밍 언어다.",
    "파이썬은 입문자에게 친절한 언어다.",          # 위와 매우 유사(중복성)
    "파이썬은 데이터 분석에 많이 쓰인다.",
    "자바는 엔터프라이즈 백엔드에 많이 쓰인다.",      # 다양성 축
]
vs = Chroma.from_documents([Document(page_content=t) for t in DOCS], embedding=embeddings)
query = "파이썬 활용"

print("=== 유사도 검색 (k=3) — 비슷한 문서가 몰릴 수 있음 ===")
for d in vs.similarity_search(query, k=3):
    print("  →", d.page_content)

print("\n=== MMR 검색 (k=3, fetch_k=4) — 다양성 확보 ===")
for d in vs.max_marginal_relevance_search(query, k=3, fetch_k=4):
    print("  →", d.page_content)

print("\n💡 MMR은 관련성과 다양성의 균형으로 중복 결과를 줄인다.")
print("✅ ch03 검색 최적화 실습 완료")
