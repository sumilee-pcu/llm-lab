"""
러닝 랭체인 — 2장 문서 로더 & 임베딩
(원본: ch2/py/*.py — a-text-loader.py ~ h-load-split-embed.py)

【교재→2026 업데이트】
  - OpenAIEmbeddings → GoogleGenerativeAIEmbeddings (models/gemini-embedding-2)
  - langchain_postgres / PGVector → Chroma (경량 환경, Docker 불필요)
  - TextLoader 경로: 절대경로 또는 더미 텍스트 생성으로 대체
  - b-web-loader.py: WebBaseLoader → 더미 문서로 대체 (네트워크 의존성 제거)
  - c-pdf-loader.py: PyPDFLoader → test.pdf 대신 더미 문서 사용
  - ✅ LangChain 1.x + Gemini Embeddings 검증 완료
"""
import os
from dotenv import load_dotenv
load_dotenv()

from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter, MarkdownHeaderTextSplitter
from langchain_community.vectorstores import Chroma

EMBEDDING_MODEL = os.getenv("GEMINI_EMBEDDING_MODEL", "models/gemini-embedding-2")
embeddings_model = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL)

# ── a. 텍스트 로더 (더미 문서) ────────────────────────────────────────────
print("=== a. 문서 준비 (더미 텍스트) ===")
raw_text = """
LangChain은 LLM 기반 애플리케이션 개발을 위한 오픈소스 프레임워크입니다.
2022년 Harrison Chase가 창립했으며, LCEL(LangChain Expression Language)을 통해
체인을 직관적으로 구성할 수 있습니다.

LangGraph는 LangChain 위에서 동작하는 그래프 기반 에이전트 프레임워크입니다.
StateGraph, 노드, 엣지를 사용하여 복잡한 멀티에이전트 워크플로를 구성합니다.
조건부 분기와 루프를 지원하여 ReAct 패턴 등을 구현할 수 있습니다.

RAG(Retrieval-Augmented Generation)는 외부 지식베이스를 검색하여
LLM의 응답 정확도를 높이는 기법입니다. 벡터 스토어에 문서를 임베딩하고
질의와 의미적으로 유사한 문서를 검색합니다.

벡터 임베딩은 텍스트를 고차원 수치 벡터로 변환합니다.
의미적으로 유사한 텍스트는 벡터 공간에서 가까운 거리에 위치합니다.
코사인 유사도, 유클리드 거리 등으로 유사도를 계산합니다.
"""
documents = [Document(page_content=raw_text, metadata={"source": "dummy.txt"})]
print(f"원본 문서: {len(documents)}개")

# ── d. 재귀 텍스트 분할기 ─────────────────────────────────────────────────
print("\n=== d. 재귀 텍스트 분할 ===")
text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=40)
split_docs = text_splitter.split_documents(documents)
print(f"분할 후: {len(split_docs)}개 청크")
for i, doc in enumerate(split_docs[:2]):
    print(f"  청크 {i}: {doc.page_content[:80]}...")

# ── f. 마크다운 분할기 ────────────────────────────────────────────────────
print("\n=== f. 마크다운 헤더 분할 ===")
md_text = """# LangChain 소개
LangChain은 LLM 기반 프레임워크입니다.

## 주요 기능
LCEL, RAG, 에이전트를 지원합니다.

## 설치 방법
pip install langchain langchain-core
"""
md_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=[
    ("#", "h1"), ("##", "h2")
])
md_docs = md_splitter.split_text(md_text)
print(f"마크다운 청크: {len(md_docs)}개")
for doc in md_docs:
    print(f"  메타데이터: {doc.metadata} | 내용: {doc.page_content[:50]}")

# ── g. 임베딩 ─────────────────────────────────────────────────────────────
print("\n=== g. 임베딩 (Gemini) ===")
texts = [
    "안녕하세요!",
    "Hello there!",
    "이름이 무엇인가요?",
    "What is your name?",
    "LangChain이란 무엇인가요?",
]
embeddings = embeddings_model.embed_documents(texts)
print(f"임베딩 수: {len(embeddings)}, 벡터 차원: {len(embeddings[0])}")

# 코사인 유사도 확인
import math
def cosine_similarity(a, b):
    dot = sum(x*y for x,y in zip(a,b))
    na = math.sqrt(sum(x*x for x in a))
    nb = math.sqrt(sum(x*x for x in b))
    return dot / (na * nb)

sim_kr_en = cosine_similarity(embeddings[0], embeddings[1])
sim_name = cosine_similarity(embeddings[2], embeddings[3])
print(f"'안녕하세요' vs 'Hello there': {sim_kr_en:.3f}")
print(f"'이름이 무엇?' vs 'What is your name?': {sim_name:.3f}")

# ── h. 로드-분할-임베딩-저장 통합 ─────────────────────────────────────────
print("\n=== h. 문서 로드→분할→임베딩→Chroma 저장 ===")
vectorstore = Chroma.from_documents(
    documents=split_docs,
    embedding=embeddings_model,
    collection_name="learning_langchain",
)
retriever = vectorstore.as_retriever(search_kwargs={"k": 2})
results = retriever.invoke("LangGraph란 무엇인가요?")
print(f"검색 결과 {len(results)}개:")
for doc in results:
    print(f"  → {doc.page_content[:80]}...")

print("\n✅ ch02 문서 로더 & 임베딩 실습 완료")
