"""
요즘 AI 에이전트 개발 — LangGraph 멀티에이전트
강의 요소: LangGraph 멀티에이전트

【교재→2026 업데이트】
  - 작성자→편집자 2-에이전트 파이프라인 (LangGraph StateGraph, Gemini)
"""
import os
from dotenv import load_dotenv
load_dotenv()

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, END
from typing import TypedDict

llm = ChatGoogleGenerativeAI(model=os.getenv("GEMINI_MODEL", "gemini-2.5-flash"), temperature=0.3)


class State(TypedDict):
    topic: str
    draft: str
    final: str


def writer(state):
    r = llm.invoke([HumanMessage(content=f"{state['topic']} 홍보 문구 1개 작성(짧게).")])
    return {"draft": r.content}


def editor(state):
    r = llm.invoke([HumanMessage(content=f"다음 문구를 더 임팩트 있게 한 줄로 다듬어줘:\n{state['draft']}")])
    return {"final": r.content}


g = StateGraph(State)
g.add_node("writer", writer)
g.add_node("editor", editor)
g.set_entry_point("writer")
g.add_edge("writer", "editor")
g.add_edge("editor", END)
app = g.compile()

out = app.invoke({"topic": "AI 학습용 노트북", "draft": "", "final": ""})
print("작성자 초안:", out["draft"].strip()[:80])
print("편집자 완성:", out["final"].strip()[:80])

print("\n✅ ch03 LangGraph 멀티에이전트 실습 완료")
