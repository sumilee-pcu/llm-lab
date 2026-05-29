"""
러닝 랭체인 — 3장 RAG
(원본: ch3/py/*.py — a-basic-rag.py ~ h-self-query.py)

【교재→2026 업데이트】
  - ChatOpenAI + OpenAIEmbeddings → ChatGoogleGenerativeAI + GoogleGenerativeAIEmbeddings
  - PGVector (Docker 필요) → Chroma (경량 환경)
  - test.txt 파일 → 인라인 더미 문서
  - b-rewrite, c-multi-query, d-rag-fusion: Gemini로 대체 검증
  - i-sql-example: SQLite로 간소화
  - ✅ LangChain 1.x 검증 완료
"""
import os
from dotenv import load_dotenv
load_dotenv()

from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, chain
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma

GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
EMBEDDING_MODEL = os.getenv("GEMINI_EMBEDDING_MODEL", "models/gemini-embedding-2")

llm = ChatGoogleGenerativeAI(model=GEMINI_MODEL, temperature=0)
embeddings_model = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL)

# ── 공통: 벡터 스토어 초기화 ─────────────────────────────────────────────
docs_text = [
    """고대 그리스 철학의 주요 인물로는 소크라테스, 플라톤, 아리스토텔레스가 있습니다.
    소크라테스(470~399 BC)는 산파술(대화법)로 유명하며, 자신은 저술을 남기지 않았습니다.
    플라톤은 소크라테스의 제자로 이데아론을 주창했으며, 아카데미아를 설립했습니다.
    아리스토텔레스는 플라톤의 제자로 논리학, 자연학, 윤리학 등 다양한 분야를 연구했습니다.""",

    """LangChain은 LLM 기반 애플리케이션 개발 프레임워크입니다.
    LCEL(LangChain Expression Language)은 체인을 파이프 연산자(|)로 연결하는 방식입니다.
    주요 컴포넌트로는 프롬프트 템플릿, 모델, 출력 파서, 검색기가 있습니다.
    LangGraph는 상태 그래프로 에이전트 워크플로를 구성하는 LangChain 확장입니다.""",

    """RAG(Retrieval-Augmented Generation)는 외부 지식을 활용하는 LLM 기법입니다.
    문서를 청크로 분할하고, 임베딩하여 벡터 스토어에 저장합니다.
    질의와 유사한 문서를 검색하여 컨텍스트로 제공합니다.
    이를 통해 LLM의 지식 한계를 극복하고 정확도를 높일 수 있습니다.""",
]

raw_documents = [Document(page_content=t) for t in docs_text]
split_docs = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50).split_documents(raw_documents)

vectorstore = Chroma.from_documents(documents=split_docs, embedding=embeddings_model)
retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# ── a. 기본 RAG ────────────────────────────────────────────────────────────
print("=== a. 기본 RAG ===")
rag_prompt = ChatPromptTemplate.from_template(
    """다음 컨텍스트만 사용해서 질문에 답하세요:
{context}

질문: {question}"""
)

# @chain 데코레이터 패턴 (교재 원본 패턴)
@chain
def qa(input: str) -> str:
    docs = retriever.invoke(input)
    formatted = rag_prompt.invoke({"context": docs, "question": input})
    answer = llm.invoke(formatted)
    return answer.content

result = qa.invoke("고대 그리스 철학의 주요 인물은 누구인가요?")
print(result[:200])

# ── b. 쿼리 재작성 (Query Rewrite) ──────────────────────────────────────
print("\n=== b. 쿼리 재작성 ===")
rewrite_prompt = ChatPromptTemplate.from_template(
    "다음 질문을 벡터 검색에 최적화된 형태로 재작성하세요. 핵심 키워드 중심으로.\n질문: {question}\n재작성:"
)
rewrite_chain = rewrite_prompt | llm | StrOutputParser()

original_query = "랭체인이 뭐야?"
rewritten = rewrite_chain.invoke({"question": original_query})
print(f"원본: {original_query}")
print(f"재작성: {rewritten}")

# 재작성된 쿼리로 RAG
result_rewritten = qa.invoke(rewritten)
print(f"답변: {result_rewritten[:150]}")

# ── c. 멀티 쿼리 (Multi-Query) ───────────────────────────────────────────
print("\n=== c. 멀티 쿼리 ===")
multi_query_prompt = ChatPromptTemplate.from_template(
    """질문을 검색 관점이 다른 3개의 변형 질문으로 만드세요.
각 질문은 줄바꿈으로 구분하세요.
질문: {question}
변형 질문:"""
)
multi_query_chain = multi_query_prompt | llm | StrOutputParser()
variants = multi_query_chain.invoke({"question": "RAG의 작동 방식은?"})
print("생성된 변형 질문들:")
for q in variants.strip().split("\n")[:3]:
    print(f"  - {q}")

# 멀티 쿼리로 검색 (중복 제거)
all_docs = []
seen = set()
for q in variants.strip().split("\n")[:3]:
    if q.strip():
        for doc in retriever.invoke(q.strip()):
            if doc.page_content not in seen:
                seen.add(doc.page_content)
                all_docs.append(doc)

print(f"멀티 쿼리 검색 결과: {len(all_docs)}개 고유 문서")
final_answer = (rag_prompt | llm | StrOutputParser()).invoke({
    "context": format_docs(all_docs),
    "question": "RAG의 작동 방식은?"
})
print(f"답변: {final_answer[:150]}")

print("\n✅ ch03 RAG 실습 완료")
