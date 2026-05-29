"""
요즘 AI 에이전트 개발 — 도구 사용 에이전트
강의 요소: OpenAI Agents SDK 에이전트

【교재→2026 업데이트】
  - OpenAI Agents SDK (Agent/Runner) → LangGraph create_react_agent (Gemini)
    : 동일한 '도구 사용 에이전트' 개념을 키 하나로 로컬 실행
"""
import os, datetime
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
def word_count(text: str) -> str:
    """문자열의 단어 수를 셉니다."""
    return str(len(text.split()))


@tool
def to_upper(text: str) -> str:
    """영문을 대문자로 바꿉니다."""
    return text.upper()


agent = create_react_agent(llm, [word_count, to_upper],
                           prompt="도구를 활용하는 에이전트입니다.")
result = agent.invoke({"messages": [
    ("human", "'hello agent world'의 단어 수를 세고, 그 문장을 대문자로도 바꿔줘")]})
print(msg_text(result["messages"][-1].content)[:150])

print("\n✅ ch02 도구 사용 에이전트 실습 완료")
