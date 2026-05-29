"""
두잇! LLM 활용 에이전트 입문 — 5장 실습
: 도구(Tool) 연동 에이전트

【교재→2026 업데이트】
  - initialize_agent() + AgentExecutor 완전 제거됨 (LangChain 1.x)
    교재: from langchain.agents import initialize_agent, AgentExecutor
    현재: from langgraph.prebuilt import create_react_agent
  - @tool 데코레이터: from langchain.tools → from langchain_core.tools
  - 실행: executor.run() → agent.invoke({"messages": [...]})
"""
import os, math
from dotenv import load_dotenv
load_dotenv()

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent

llm = ChatGoogleGenerativeAI(model=os.getenv("GEMINI_MODEL", "gemini-2.5-flash"), temperature=0)

def msg_text(content) -> str:
    """Gemini 2.5는 응답을 list[dict] (thought signature 포함) 형태로 줄 때가 있어 문자열로 평탄화한다."""
    if isinstance(content, list):
        return " ".join(c.get("text", "") for c in content if isinstance(c, dict)).strip()
    return str(content)

@tool
def calculator(expression: str) -> str:
    """수학 계산을 수행합니다. 파이썬 수식을 입력받아 결과를 반환합니다."""
    try:
        result = eval(expression, {"math": math, "__builtins__": {}})
        return f"계산 결과: {result}"
    except Exception as e:
        return f"계산 오류: {e}"

@tool
def get_weather(city: str) -> str:
    """도시의 현재 날씨를 반환합니다. (실습용 더미 데이터)"""
    weather_db = {"서울": "맑음, 23°C", "부산": "흐림, 20°C", "제주": "비, 18°C"}
    return weather_db.get(city, f"{city}: 날씨 정보 없음")

agent = create_react_agent(
    llm, [calculator, get_weather],
    prompt="당신은 도움이 되는 AI 어시스턴트입니다. 필요시 도구를 사용하세요."
)

result = agent.invoke({"messages": [("human", "서울 날씨 알려주고 144의 제곱근도 계산해줘")]})
print(msg_text(result["messages"][-1].content))

print("\n✅ ch05 도구(Tool) 연동 에이전트 실습 완료")
