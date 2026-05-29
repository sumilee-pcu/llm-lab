"""
실전 LLM (한빛, 2판) — 의미 검색 (semantic search)
강의 요소: 의미 검색

【교재→2026 업데이트】
  - OpenAIEmbeddings → GoogleGenerativeAIEmbeddings (models/gemini-embedding-2)
  - 키워드 매칭이 아닌 임베딩 유사도 기반 검색 시연
"""
import os, math
from dotenv import load_dotenv
load_dotenv()

from langchain_google_genai import GoogleGenerativeAIEmbeddings

embeddings = GoogleGenerativeAIEmbeddings(model=os.getenv("GEMINI_EMBEDDING_MODEL", "models/gemini-embedding-2"))


def cosine(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    return dot / (math.sqrt(sum(x*x for x in a)) * math.sqrt(sum(y*y for y in b)))


corpus = [
    "파이썬으로 웹 크롤러 만드는 법",
    "맛있는 김치찌개 끓이는 레시피",
    "딥러닝 모델 학습 속도 높이기",
    "서울 근교 가볼 만한 등산 코스",
]
doc_vecs = embeddings.embed_documents(corpus)

query = "신경망 훈련을 빠르게 하려면?"   # 키워드는 안 겹치지만 의미는 '딥러닝 학습 속도'
q_vec = embeddings.embed_query(query)

ranked = sorted(((cosine(q_vec, v), doc) for v, doc in zip(doc_vecs, corpus)), reverse=True)
print(f"질의: {query}\n")
print("의미 검색 순위:")
for score, doc in ranked:
    print(f"  {score:.3f}  {doc}")

print("\n✅ ch02 의미 검색 실습 완료")
