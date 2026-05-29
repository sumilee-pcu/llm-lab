"""
실전 LLM 앱 개발 — 7장 LangChain RAG 실습
(원본: 7_langchain/rag/vector_search_upstage.ipynb → Gemini Embeddings로 대체)

【교재→2026 업데이트】
  - Upstage Solar Embeddings → GoogleGenerativeAIEmbeddings (gemini-embedding-2)
  - ChatOpenAI / ChatUpstage → ChatGoogleGenerativeAI
  - langchain==0.2.x → 1.x (RetrievalQA 제거됨)
    구 방식: RetrievalQA.from_chain_type(llm, retriever=...)
    신 방식: retriever | prompt | llm | parser (LCEL)
  - chromadb → 동일 (0.5.x 호환)
  - 외부 문서 의존성 제거 → 인라인 텍스트 문서 사용 (경량 환경)
  - ✅ LangChain 1.x + Gemini Embeddings 검증 완료
"""
import os
from dotenv import load_dotenv
load_dotenv()

from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma

GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
EMBEDDING_MODEL = os.getenv("GEMINI_EMBEDDING_MODEL", "models/gemini-embedding-2")

llm = ChatGoogleGenerativeAI(model=GEMINI_MODEL, temperature=0)
embeddings = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL)

# ── 1. 문서 준비 (인라인 더미 문서) ─────────────────────────────────────
print("=== 1. 문서 준비 ===")
docs_text = [
    """LangChain은 LLM 기반 애플리케이션 개발을 위한 오픈소스 프레임워크입니다.
    체인(Chain), 에이전트(Agent), 메모리(Memory) 등의 핵심 컴포넌트를 제공합니다.
    LCEL(LangChain Expression Language)을 통해 파이프라인을 직관적으로 구성할 수 있습니다.""",

    """LangGraph는 LangChain 팀이 개발한 그래프 기반 에이전트 프레임워크입니다.
    StateGraph를 사용하여 복잡한 멀티에이전트 워크플로를 구성할 수 있습니다.
    노드(Node)와 엣지(Edge)로 에이전트의 동작 흐름을 명시적으로 정의합니다.""",

    """RAG(Retrieval-Augmented Generation)는 외부 지식베이스를 활용해
    LLM의 응답 정확도를 높이는 기법입니다.
    벡터 데이터베이스에 문서를 저장하고, 질의와 유사한 문서를 검색해 컨텍스트로 제공합니다.""",

    """Gemini는 Google DeepMind가 개발한 멀티모달 AI 모델입니다.
    텍스트, 이미지, 오디오, 동영상을 이해하고 생성할 수 있습니다.
    gemini-2.5-flash는 빠른 추론 속도와 비용 효율성이 장점입니다.""",

    """벡터 임베딩은 텍스트를 고차원 수치 벡터로 변환하는 기법입니다.
    의미적으로 유사한 텍스트는 벡터 공간에서 가까운 거리에 위치합니다.
    RAG 시스템에서 문서 검색의 핵심 기술로 활용됩니다.""",
]

documents = [Document(page_content=t) for t in docs_text]
splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=30)
split_docs = splitter.split_documents(documents)
print(f"원본 문서: {len(documents)}개 → 분할 후: {len(split_docs)}개 청크")

# ── 2. 벡터 스토어 생성 ───────────────────────────────────────────────────
print("\n=== 2. 벡터 스토어 생성 (Chroma + Gemini Embeddings) ===")
vectorstore = Chroma.from_documents(
    documents=split_docs,
    embedding=embeddings,
    collection_name="llm_docs",
)
retriever = vectorstore.as_retriever(search_kwargs={"k": 2})
print("벡터 스토어 생성 완료")

# ── 3. RAG 체인 구성 (LCEL) ──────────────────────────────────────────────
print("\n=== 3. RAG 체인 (LCEL 방식) ===")
# 구 방식: RetrievalQA.from_chain_type(llm, retriever=...) — LangChain 1.x에서 제거됨
# 신 방식: LCEL 파이프라인

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

rag_prompt = ChatPromptTemplate.from_messages([
    ("system", """당신은 문서 기반 Q&A 어시스턴트입니다.
아래 컨텍스트만 사용해서 답변하세요. 컨텍스트에 없는 내용은 '문서에 없는 내용입니다'라고 하세요.

컨텍스트:
{context}"""),
    ("human", "{question}"),
])

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | rag_prompt
    | llm
    | StrOutputParser()
)

# ── 4. RAG 질의 테스트 ────────────────────────────────────────────────────
print("\n=== 4. RAG 질의 테스트 ===")
questions = [
    "LangGraph가 무엇인가요?",
    "RAG 시스템에서 벡터 임베딩은 어떻게 활용되나요?",
    "gemini-2.5-flash의 장점은?",
]

for q in questions:
    print(f"\nQ: {q}")
    answer = rag_chain.invoke(q)
    print(f"A: {answer[:200]}")

print("\n✅ ch07 LangChain RAG 실습 완료")
