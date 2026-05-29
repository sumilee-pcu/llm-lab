"""
Gemini 임베딩 모델 차원 진단 스크립트.

OpenAI API 비용 발생을 막기 위해 Google Gemini 임베딩만 확인합니다.
"""

import os

from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma

from provider_config import get_gemini_embeddings


load_dotenv()

print("=" * 60)
print("Gemini 임베딩 모델 차원 진단")
print("=" * 60)

try:
    embeddings = get_gemini_embeddings()
    vector = embeddings.embed_query("test")
    print(f"차원: {len(vector)}")
    print(f"모델: {embeddings.model}")
except Exception as e:
    print(f"Gemini 임베딩 호출 오류: {e}")

print("\n" + "=" * 60)
print("기존 ChromaDB 확인")
print("=" * 60)

for db_name in ["experience_db", "experience_db_v2", "medical_knowledge_db"]:
    if not os.path.exists(db_name):
        print(f"\n[{db_name}] 폴더가 존재하지 않습니다.")
        continue

    try:
        vs = Chroma(
            persist_directory=db_name,
            embedding_function=embeddings,
            collection_name="code_refactoring_experiences",
        )
        count = vs._collection.count()
        print(f"\n[{db_name}]")
        print(f"문서 개수: {count}")
        if count > 0:
            result = vs._collection.get(limit=1, include=["embeddings"])
            if result["embeddings"] and len(result["embeddings"]) > 0:
                print(f"저장된 임베딩 차원: {len(result['embeddings'][0])}")
    except Exception as e:
        print(f"\n[{db_name}] 읽기 오류: {e}")

print("\n해결 방법:")
print("기존 DB와 Gemini 임베딩 차원이 맞지 않으면 해당 실습 DB 폴더를 삭제하고 다시 색인하세요.")
