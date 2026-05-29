"""
이것이 멀티 에이전트다 (한빛) — 싱글 에이전트 ①: 영업 데이터 분석
강의 요소: 싱글 에이전트 — 영업 데이터 분석·집계

【교재→2026 업데이트】
  - OpenAI/Cursor 기반 → LangGraph create_react_agent (Gemini) + 분석 도구
  - 외부 의존 없이 인라인 데이터로 집계 도구 호출 시연
"""
import os
from dotenv import load_dotenv
load_dotenv()

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent

llm = ChatGoogleGenerativeAI(model=os.getenv("GEMINI_MODEL", "gemini-2.5-flash"), temperature=0)

SALES = [{"region": "서울", "amount": 1200}, {"region": "부산", "amount": 800},
         {"region": "서울", "amount": 1300}, {"region": "대구", "amount": 500}]


def msg_text(content) -> str:
    if isinstance(content, list):
        return " ".join(c.get("text", "") for c in content if isinstance(c, dict)).strip()
    return str(content)


@tool
def total_by_region(region: str) -> str:
    """특정 지역의 총 매출을 집계합니다."""
    return str(sum(s["amount"] for s in SALES if s["region"] == region))


@tool
def top_region() -> str:
    """매출이 가장 높은 지역을 반환합니다."""
    agg = {}
    for s in SALES:
        agg[s["region"]] = agg.get(s["region"], 0) + s["amount"]
    return max(agg, key=agg.get)


agent = create_react_agent(llm, [total_by_region, top_region],
                           prompt="영업 데이터 분석 에이전트입니다. 도구로 집계해 답하세요.")
result = agent.invoke({"messages": [("human", "서울 총매출과, 매출 1위 지역을 알려줘")]})
print(msg_text(result["messages"][-1].content)[:150])

print("\n✅ ch01 싱글 에이전트(영업분석) 실습 완료")
