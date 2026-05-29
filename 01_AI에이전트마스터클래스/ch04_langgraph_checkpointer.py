"""
AI 에이전트 마스터 클래스 (한빛) — LangGraph 체크포인터로 대화 상태 저장
강의 요소: LangGraph 체크포인터 / (MCP·Streamlit은 개념 — 별도 환경 필요)

【교재→2026 업데이트】
  - LangGraph 1.2.x: MemorySaver 체크포인터 + thread_id 로 세션별 상태 유지
  - create_react_agent(checkpointer=...) 패턴 (langgraph.prebuilt)
"""
import os
from dotenv import load_dotenv
load_dotenv()

from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver

llm = ChatGoogleGenerativeAI(model=os.getenv("GEMINI_MODEL", "gemini-2.5-flash"), temperature=0)


def msg_text(content) -> str:
    if isinstance(content, list):
        return " ".join(c.get("text", "") for c in content if isinstance(c, dict)).strip()
    return str(content)


# 체크포인터(메모리) — thread_id 별로 대화 상태 자동 저장/복원
checkpointer = MemorySaver()
agent = create_react_agent(llm, [], checkpointer=checkpointer)

config = {"configurable": {"thread_id": "user-wine-session"}}

print("=== 같은 thread_id 로 상태 유지 ===")
r1 = agent.invoke({"messages": [("human", "내가 좋아하는 와인은 피노 누아야.")]}, config)
print("A1:", msg_text(r1["messages"][-1].content)[:80])

# 같은 thread_id → 이전 대화 기억
r2 = agent.invoke({"messages": [("human", "내가 좋아한다고 한 와인이 뭐였지?")]}, config)
print("A2:", msg_text(r2["messages"][-1].content)[:80])

# 다른 thread_id → 상태 분리(기억 못 함)
r3 = agent.invoke({"messages": [("human", "내가 좋아하는 와인이 뭐야?")]},
                  {"configurable": {"thread_id": "other-session"}})
print("A3(다른 세션):", msg_text(r3["messages"][-1].content)[:80])

print("\n✅ ch04 LangGraph 체크포인터 상태 저장 실습 완료")
