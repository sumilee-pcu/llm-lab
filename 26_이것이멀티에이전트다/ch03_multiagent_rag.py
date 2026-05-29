"""
이것이 멀티 에이전트다 (한빛) — 멀티 에이전트: 약관 기반 RAG 질의응답
강의 요소: 멀티 — 약관 RAG 질의응답 (검색 에이전트 + 답변 에이전트)

【교재→2026 업데이트】
  - 검색 노드(retriever) → 답변 노드(LLM) 2단 LangGraph 파이프
  - GoogleGenerativeAIEmbeddings + Chroma 인메모리 (약관 텍스트 인라인)
"""
import os
from dotenv import load_dotenv
load_dotenv()

from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, END
from typing import TypedDict

llm = ChatGoogleGenerativeAI(model=os.getenv("GEMINI_MODEL", "gemini-2.5-flash"), temperature=0)
embeddings = GoogleGenerativeAIEmbeddings(model=os.getenv("GEMINI_EMBEDDING_MODEL", "models/gemini-embedding-2"))

TERMS = ["제3조 환불: 디지털 콘텐츠는 다운로드 전 7일 이내 환불 가능합니다.",
         "제5조 책임: 회사는 천재지변으로 인한 손해를 책임지지 않습니다.",
         "제8조 해지: 회원은 언제든 탈퇴할 수 있으며 즉시 처리됩니다."]
retriever = Chroma.from_documents([Document(page_content=t) for t in TERMS],
                                  embedding=embeddings).as_retriever(search_kwargs={"k": 1})


class State(TypedDict):
    question: str
    context: str
    answer: str


def search_agent(state):                      # 검색 에이전트
    docs = retriever.invoke(state["question"])
    return {"context": "\n".join(d.page_content for d in docs)}


def answer_agent(state):                       # 답변 에이전트
    r = llm.invoke([HumanMessage(content=(
        f"약관 근거로만 답하세요.\n[약관]\n{state['context']}\n[질문] {state['question']}"))])
    return {"answer": r.content}


g = StateGraph(State)
g.add_node("search", search_agent)
g.add_node("answer", answer_agent)
g.set_entry_point("search")
g.add_edge("search", "answer")
g.add_edge("answer", END)
app = g.compile()

out = app.invoke({"question": "디지털 콘텐츠 환불 조건이 어떻게 돼?", "context": "", "answer": ""})
print("검색된 약관:", out["context"][:60])
print("답변:", out["answer"].strip()[:140])

print("\n✅ ch03 멀티에이전트 약관 RAG 실습 완료")
