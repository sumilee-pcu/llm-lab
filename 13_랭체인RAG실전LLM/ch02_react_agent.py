"""
랭체인과 RAG로 배우는 실전 LLM (위키북스) — ReAct 추론·행동 에이전트
강의 요소: ReAct 추론·행동 에이전트

【교재→2026 업데이트】
  - initialize_agent(ZERO_SHOT_REACT) → create_react_agent (langgraph.prebuilt)
  - 추론(Reasoning)→도구 행동(Acting) 반복을 도구 다단 호출로 시연
"""
import os, math
from dotenv import load_dotenv
load_dotenv()

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent

llm = ChatGoogleGenerativeAI(model=os.getenv("GEMINI_MODEL", "gemini-2.5-flash"), temperature=0)


def msg_text(content) -> str:
    if isinstance(content, list):
        return " ".join(c.get("text", "") for c in content if isinstance(c, dict)).strip()
    return str(content)


@tool
def population(country: str) -> str:
    """국가의 인구(백만 명)를 반환합니다. (실습용 더미)"""
    db = {"한국": 51, "일본": 124, "미국": 333}
    return str(db.get(country, "정보 없음"))


@tool
def multiply(a: float, b: float) -> str:
    """두 수를 곱합니다."""
    return str(a * b)


# 여러 도구를 번갈아 써야 풀리는 질문 → ReAct 추론·행동 반복
agent = create_react_agent(llm, [population, multiply],
                           prompt="도구로 단계적으로 추론하는 ReAct 에이전트입니다.")
result = agent.invoke({"messages": [
    ("human", "한국과 일본 인구를 각각 찾아서 더하면 몇 백만 명이야? 도구를 사용해 계산해줘.")
]})
print(msg_text(result["messages"][-1].content)[:200])
print(f"\n(에이전트가 호출한 메시지 단계 수: {len(result['messages'])})")

print("\n✅ ch02 ReAct 에이전트 실습 완료")
