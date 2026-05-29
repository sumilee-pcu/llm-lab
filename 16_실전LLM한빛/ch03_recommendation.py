"""
실전 LLM (한빛, 2판) — 임베딩 기반 추천 시스템
강의 요소: 추천 시스템
(파인튜닝·양자화·VQA·RLHF는 GPU/멀티모달 필요 → 강의 슬라이드로 설명, 코드는 추천 시연)

【교재→2026 업데이트】
  - 아이템 임베딩 유사도로 콘텐츠 기반 추천 (GoogleGenerativeAIEmbeddings)
"""
import os, math
from dotenv import load_dotenv
load_dotenv()

from langchain_google_genai import GoogleGenerativeAIEmbeddings

embeddings = GoogleGenerativeAIEmbeddings(model=os.getenv("GEMINI_EMBEDDING_MODEL", "models/gemini-embedding-2"))


def cosine(a, b):
    return sum(x*y for x, y in zip(a, b)) / (
        math.sqrt(sum(x*x for x in a)) * math.sqrt(sum(y*y for y in b)))


movies = {
    "인터스텔라": "우주 여행과 시간 왜곡을 다룬 SF 대작",
    "그래비티": "우주에서 조난당한 우주인의 생존 SF",
    "노트북": "오랜 시간을 함께한 연인의 로맨스 드라마",
    "어벤져스": "슈퍼히어로들이 지구를 지키는 액션",
}
titles = list(movies)
vecs = {t: embeddings.embed_query(movies[t]) for t in titles}


def recommend(liked: str, top: int = 2):
    base = vecs[liked]
    scored = sorted(((cosine(base, vecs[t]), t) for t in titles if t != liked), reverse=True)
    return scored[:top]


for liked in ["인터스텔라", "노트북"]:
    print(f"'{liked}'를 좋아한 사용자 추천:")
    for score, t in recommend(liked):
        print(f"  {score:.3f}  {t} — {movies[t]}")
    print()

print("✅ ch03 추천 시스템 실습 완료")
