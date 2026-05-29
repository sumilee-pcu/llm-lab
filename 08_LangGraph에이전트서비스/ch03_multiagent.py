"""
LangGraph로 만드는 AI 에이전트 서비스 — 멀티 에이전트 설계·협업
강의 요소: 멀티 에이전트 설계·협업 (Supervisor → Worker)

【교재→2026 업데이트】
  - 슈퍼바이저가 작업을 분류해 적절한 워커 노드로 라우팅 (LangGraph 1.2.x)
"""
import os
from dotenv import load_dotenv
load_dotenv()

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import StateGraph, END
from typing import TypedDict, Literal

llm = ChatGoogleGenerativeAI(model=os.getenv("GEMINI_MODEL", "gemini-2.5-flash"), temperature=0)


class State(TypedDict):
    task: str
    worker: str
    result: str


supervisor_chain = ChatPromptTemplate.from_template(
    "작업을 'coding'/'writing'/'research' 중 하나로만 분류: {task}"
) | llm | StrOutputParser()


def supervisor(state):
    label = supervisor_chain.invoke({"task": state["task"]}).strip().lower()
    for w in ("coding", "writing", "research"):
        if w in label:
            return {"worker": w}
    return {"worker": "research"}


def make_worker(role):
    def node(state):
        r = llm.invoke([HumanMessage(content=f"당신은 {role} 전문가. 한 문장으로: {state['task']}")])
        return {"result": f"[{role}] {r.content[:90]}"}
    return node


def route(state) -> Literal["coding", "writing", "research"]:
    return state["worker"]


g = StateGraph(State)
g.add_node("supervisor", supervisor)
g.add_node("coding", make_worker("코딩"))
g.add_node("writing", make_worker("글쓰기"))
g.add_node("research", make_worker("리서치"))
g.set_entry_point("supervisor")
g.add_conditional_edges("supervisor", route,
                        {"coding": "coding", "writing": "writing", "research": "research"})
for w in ("coding", "writing", "research"):
    g.add_edge(w, END)
app = g.compile()

for task in ["파이썬으로 퀵소트 구현해줘", "신제품 홍보 문구 써줘", "전기차 시장 동향 알려줘"]:
    out = app.invoke({"task": task, "worker": "", "result": ""})
    print(f"작업: {task}\n  배정: {out['worker']} | {out['result']}\n")

print("✅ ch03 멀티 에이전트 협업 실습 완료")
