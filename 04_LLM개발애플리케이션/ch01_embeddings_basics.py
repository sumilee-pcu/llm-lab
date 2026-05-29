"""
LLM 개발 애플리케이션 — 임베딩 기초 (트랜스포머/BERT/GPT 이론의 실습 대표)
강의 요소: 임베딩 / 어텐션·의미 표현

【교재→2026 업데이트】
  - OpenAIEmbeddings → GoogleGenerativeAIEmbeddings (models/gemini-embedding-2)
  - 트랜스포머/BERT/GPT 이론은 강의 슬라이드, 임베딩은 실제 벡터로 체험
"""
import os, math
from dotenv import load_dotenv
load_dotenv()

from langchain_google_genai import GoogleGenerativeAIEmbeddings

embeddings = GoogleGenerativeAIEmbeddings(model=os.getenv("GEMINI_EMBEDDING_MODEL", "models/gemini-embedding-2"))


def cosine(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a)); nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb)


sentences = [
    "고양이가 소파에서 잠을 잔다.",
    "강아지가 마당에서 뛰논다.",      # 의미 유사(동물)
    "트랜스포머는 어텐션 기반 모델이다.",  # 의미 무관
]
vecs = embeddings.embed_documents(sentences)
print(f"임베딩 차원: {len(vecs[0])}")
print(f"유사(동물 vs 동물):  {cosine(vecs[0], vecs[1]):.3f}")
print(f"무관(동물 vs 모델):  {cosine(vecs[0], vecs[2]):.3f}")

# 쿼리-문서 의미 검색
q = embeddings.embed_query("애완동물이 노는 모습")
ranked = sorted(((cosine(q, v), s) for v, s in zip(vecs, sentences)), reverse=True)
print("\n쿼리와 가장 가까운 문장:")
for score, s in ranked:
    print(f"  {score:.3f}  {s}")

print("\n✅ ch01 임베딩 기초 실습 완료")
