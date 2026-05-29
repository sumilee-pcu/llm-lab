"""
러닝 랭체인 — 4장 메모리 & 상태 관리
(원본: ch4/py/*.py — a-simple-memory.py ~ f-merge-messages.py)

【교재→2026 업데이트】
  - ChatOpenAI → ChatGoogleGenerativeAI
  - langgraph-checkpoint-sqlite: 동일 (하위 호환 유지)
  - c-persistent-memory.py: SQLite 체크포인터 → MemorySaver로 대체 (경량)
  - d-trim-messages, e-filter-messages, f-merge-messages: langchain_core.messages 동일
  - ✅ LangChain 1.x + LangGraph 1.2.x 검증 완료
"""
import os
from dotenv import load_dotenv
load_dotenv()

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, trim_messages, filter_messages
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from typing import Annotated, TypedDict

GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
model = ChatGoogleGenerativeAI(model=GEMINI_MODEL, temperature=0)

# ── a. 단순 메모리 (메시지 리스트 직접 관리) ──────────────────────────────
print("=== a. 단순 메모리 (메시지 리스트) ===")
history = []

def chat(user_input: str) -> str:
    history.append(HumanMessage(content=user_input))
    response = model.invoke(history)
    history.append(AIMessage(content=response.content))
    return response.content

print(f"Q: LangChain이란?")
print(f"A: {chat('LangChain이란?')[:100]}")
print(f"\nQ: 그 창업자는 누구야?")
print(f"A: {chat('그 창업자는 누구야?')[:100]}")

# ── b. LangGraph StateGraph 메모리 ───────────────────────────────────────
print("\n=== b. LangGraph StateGraph 메모리 ===")

class State(TypedDict):
    messages: Annotated[list, add_messages]

def call_model(state: State) -> State:
    response = model.invoke(state["messages"])
    return {"messages": response}

builder = StateGraph(State)
builder.add_node("model", call_model)
builder.add_edge(START, "model")
builder.add_edge("model", END)
graph = builder.compile()

result = graph.invoke({"messages": [HumanMessage(content="파이썬이란?")]})
print(result["messages"][-1].content[:100])

# ── c. 영속 메모리 (MemorySaver 체크포인터) ──────────────────────────────
print("\n=== c. 영속 메모리 (MemorySaver) ===")
checkpointer = MemorySaver()
persistent_graph = builder.compile(checkpointer=checkpointer)

config = {"configurable": {"thread_id": "session-1"}}
r1 = persistent_graph.invoke({"messages": [HumanMessage("내 이름은 수미야.")]}, config)
print(f"턴1: {r1['messages'][-1].content[:80]}")

r2 = persistent_graph.invoke({"messages": [HumanMessage("내 이름이 뭐야?")]}, config)
print(f"턴2 (이전 컨텍스트 유지): {r2['messages'][-1].content[:80]}")

# ── d. 메시지 트리밍 ─────────────────────────────────────────────────────
print("\n=== d. 메시지 트리밍 ===")
long_history = [
    SystemMessage(content="당신은 AI 어시스턴트입니다."),
    HumanMessage(content="첫 번째 질문입니다."),
    AIMessage(content="첫 번째 답변입니다."),
    HumanMessage(content="두 번째 질문입니다."),
    AIMessage(content="두 번째 답변입니다."),
    HumanMessage(content="최신 질문입니다."),
]
trimmed = trim_messages(
    long_history,
    max_tokens=100,
    strategy="last",
    token_counter=len,
    include_system=True,
)
print(f"원본 메시지 {len(long_history)}개 → 트리밍 후 {len(trimmed)}개")

# ── e. 메시지 필터링 ─────────────────────────────────────────────────────
print("\n=== e. 메시지 필터링 ===")
mixed_messages = [
    SystemMessage(content="시스템 지시사항"),
    HumanMessage(content="사용자 메시지 1"),
    AIMessage(content="AI 답변 1"),
    HumanMessage(content="사용자 메시지 2"),
]
human_only = filter_messages(mixed_messages, include_types=["human"])
print(f"사람 메시지만: {len(human_only)}개")
for m in human_only:
    print(f"  - {m.content}")

print("\n✅ ch04 메모리 & 상태 관리 실습 완료")
