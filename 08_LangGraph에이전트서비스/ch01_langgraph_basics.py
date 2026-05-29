"""
LangGraph로 만드는 AI 에이전트 서비스 — LangGraph 기초
강의 요소: LangChain 기초 / LangGraph 그래프 기반 에이전트 설계

【교재→2026 업데이트】
  - ChatOpenAI → ChatGoogleGenerativeAI (gemini-2.5-flash)
  - StateGraph / add_node / add_edge — LangGraph 1.2.x 동일
"""
import os
from dotenv import load_dotenv
load_dotenv()

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, END
from typing import TypedDict

llm = ChatGoogleGenerativeAI(model=os.getenv("GEMINI_MODEL", "gemini-2.5-flash"), temperature=0)


class State(TypedDict):
    topic: str
    draft: str
    review: str


def write_node(state):
    r = llm.invoke([HumanMessage(content=f"{state['topic']}에 대해 2문장 초안 작성.")])
    return {"draft": r.content}


def review_node(state):
    r = llm.invoke([HumanMessage(content=f"다음 초안을 한 문장으로 더 명확히 다듬어줘:\n{state['draft']}")])
    return {"review": r.content}


g = StateGraph(State)
g.add_node("write", write_node)
g.add_node("review", review_node)
g.set_entry_point("write")
g.add_edge("write", "review")
g.add_edge("review", END)
app = g.compile()

result = app.invoke({"topic": "LangGraph의 장점", "draft": "", "review": ""})
print("초안:", result["draft"].strip()[:100])
print("\n검토본:", result["review"].strip()[:100])

print("\n✅ ch01 LangGraph 기초 실습 완료")
