"""
테디노트 RAG 비법노트 — 기본편 — 2. 문서 로더·분할·임베딩
(원본 저장소: github.com/teddylee777/langchain-kr — CH04 문서 로더 / CH05 분할 / CH06 임베딩 대응)

【교재→2026 업데이트】
  - OpenAIEmbeddings → GoogleGenerativeAIEmbeddings (models/gemini-embedding-2)
  - PDF/웹 로더(외부 의존) → 인라인 더미 문서 (네트워크/파일 의존성 제거, 경량 실행)
  - RecursiveCharacterTextSplitter: langchain_text_splitters (1.x 경로)
"""
import os
from dotenv import load_dotenv
load_dotenv()

from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

EMBEDDING_MODEL = os.getenv("GEMINI_EMBEDDING_MODEL", "models/gemini-embedding-2")
embeddings = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL)

# 1. 문서 준비 (교재의 PDF 로더 대신 인라인 텍스트)
raw_text = (
    "RAG는 검색 증강 생성을 의미합니다. 외부 지식을 검색해 LLM 답변의 정확도를 높입니다. "
    "8단계 RAG 파이프라인은 문서 로드, 분할, 임베딩, 벡터 저장, 검색, 프롬프트, LLM, 체인으로 구성됩니다. "
    "LangChain은 이 파이프라인을 LCEL로 간결하게 연결합니다. "
    "테디노트 RAG 비법노트는 이 과정을 단계별로 설명합니다."
)
docs = [Document(page_content=raw_text, metadata={"source": "inline-dummy"})]

# 2. 분할 (청킹)
print("=== 1. 문서 분할 ===")
splitter = RecursiveCharacterTextSplitter(chunk_size=60, chunk_overlap=10)
split_docs = splitter.split_documents(docs)
print(f"청크 수: {len(split_docs)}")
for i, d in enumerate(split_docs):
    print(f"  [{i}] {d.page_content[:40]}...")

# 3. 임베딩
print("\n=== 2. 임베딩 ===")
texts = [d.page_content for d in split_docs]
vectors = embeddings.embed_documents(texts)
print(f"임베딩 개수: {len(vectors)}, 벡터 차원: {len(vectors[0])}")

# 4. 쿼리 임베딩 & 코사인 유사도
print("\n=== 3. 쿼리-청크 유사도 ===")
import math
def cosine(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a)); nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb)

q_vec = embeddings.embed_query("RAG 8단계 파이프라인이 뭐야?")
sims = sorted(((cosine(q_vec, v), i) for i, v in enumerate(vectors)), reverse=True)
top_score, top_idx = sims[0]
print(f"가장 유사한 청크 [{top_idx}] (유사도 {top_score:.3f}): {split_docs[top_idx].page_content[:50]}")

print("\n✅ ch02 문서 로더·분할·임베딩 실습 완료")
