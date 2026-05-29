"""
테디노트 RAG 비법노트 — 심화편 — PART04 에이전트(Agentic RAG)
강의 요소: 에이전트·Agentic RAG (검색을 '도구'로 쓰고 스스로 판단)

【교재→2026 업데이트】
  - create_react_agent + 검색 도구 → 에이전트가 필요 시에만 검색 호출
  - 단순 정보(인사 등)는 검색 없이 답, 지식 질문은 retrieve 도구 사용
"""
import os
from dotenv import load_dotenv
load_dotenv()

from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent

llm = ChatGoogleGenerativeAI(model=os.getenv("GEMINI_MODEL", "gemini-2.5-flash"), temperature=0)
embeddings = GoogleGenerativeAIEmbeddings(model=os.getenv("GEMINI_EMBEDDING_MODEL", "models/gemini-embedding-2"))


def msg_text(content) -> str:
    if isinstance(content, list):
        return " ".join(c.get("text", "") for c in content if isinstance(c, dict)).strip()
    return str(content)


DOCS = ["Agentic RAG는 에이전트가 언제 검색할지 스스로 결정하는 RAG입니다.",
        "self-RAG는 검색 결과의 유용성을 모델이 자체 평가합니다.",
        "corrective RAG는 검색 품질이 낮으면 재검색·웹검색으로 보정합니다."]
retriever = Chroma.from_documents([Document(page_content=t) for t in DOCS],
                                  embedding=embeddings).as_retriever(search_kwargs={"k": 2})


@tool
def retrieve(query: str) -> str:
    """RAG 지식베이스에서 관련 문서를 검색합니다. 지식 질문에만 사용하세요."""
    return "\n".join(d.page_content for d in retriever.invoke(query))


agent = create_react_agent(llm, [retrieve],
                           prompt="필요할 때만 retrieve 도구로 검색하는 Agentic RAG 어시스턴트입니다.")

for q in ["안녕? 넌 누구야?", "corrective RAG가 뭐야?"]:
    out = agent.invoke({"messages": [("human", q)]})
    print(f"Q: {q}\nA: {msg_text(out['messages'][-1].content)[:120]}\n")

print("✅ ch03 Agentic RAG 실습 완료")
