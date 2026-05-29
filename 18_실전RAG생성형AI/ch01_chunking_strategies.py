"""
실전 RAG 기반 생성형 AI 개발 — 청킹 전략 비교
강의 요소: 벡터 저장소·청킹 전략·색인

【교재→2026 업데이트】
  - 청크 크기/오버랩이 검색 단위에 미치는 영향을 직접 비교 (LangChain 1.x text-splitters)
  - 임베딩 의존 없는 순수 분할 비교라 빠르게 실행
"""
from langchain_text_splitters import RecursiveCharacterTextSplitter, CharacterTextSplitter
from langchain_core.documents import Document

TEXT = (
    "RAG는 검색 증강 생성이다. 청킹은 문서를 작은 단위로 나누는 과정이다. "
    "청크가 너무 크면 노이즈가 늘고, 너무 작으면 맥락이 끊긴다. "
    "오버랩은 인접 청크 간 경계 맥락 손실을 줄인다. "
    "색인은 청크 임베딩을 벡터 저장소에 저장하는 단계다."
)
docs = [Document(page_content=TEXT)]

print("=== 1. RecursiveCharacterTextSplitter (size=40, overlap=10) ===")
r = RecursiveCharacterTextSplitter(chunk_size=40, chunk_overlap=10).split_documents(docs)
print(f"청크 수: {len(r)}")
for i, d in enumerate(r):
    print(f"  [{i}] ({len(d.page_content)}자) {d.page_content[:35]}")

print("\n=== 2. 큰 청크 (size=80, overlap=0) ===")
big = RecursiveCharacterTextSplitter(chunk_size=80, chunk_overlap=0).split_documents(docs)
print(f"청크 수: {len(big)} → 청크가 적고 큼(맥락↑ 정밀도↓)")

print("\n=== 3. 문장 단위 (구분자 '. ') ===")
sent = CharacterTextSplitter(separator=". ", chunk_size=30, chunk_overlap=0).split_documents(docs)
print(f"청크 수: {len(sent)}")
for d in sent:
    print(f"  - {d.page_content[:40]}")

print("\n💡 결론: chunk_size·overlap·구분자 선택이 검색 품질을 좌우한다.")
print("✅ ch01 청킹 전략 실습 완료")
