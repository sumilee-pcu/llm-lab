"""
이것이 멀티 에이전트다 (한빛) — 싱글 에이전트 ②: 개인정보 탐지·마스킹
강의 요소: 싱글 에이전트 — 개인정보 탐지·마스킹

【교재→2026 업데이트】
  - 정규식 기반 PII 탐지 도구 + LLM 에이전트 결합 (Gemini)
  - 전화/이메일/주민번호 패턴 마스킹 (외부 의존 없음)
"""
import os, re
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
def mask_pii(text: str) -> str:
    """텍스트에서 이메일·전화번호·주민등록번호를 찾아 마스킹합니다."""
    text = re.sub(r"[\w.+-]+@[\w-]+\.[\w.-]+", "[이메일]", text)
    text = re.sub(r"01[016789]-?\d{3,4}-?\d{4}", "[전화번호]", text)
    text = re.sub(r"\d{6}-?\d{7}", "[주민번호]", text)
    return text


agent = create_react_agent(llm, [mask_pii],
                           prompt="개인정보 보호 에이전트입니다. mask_pii 도구로 민감정보를 가린 뒤 안전한 텍스트를 보여주세요.")
sample = "고객 김철수, 이메일 chulsoo@example.com, 연락처 010-1234-5678 입니다."
result = agent.invoke({"messages": [("human", f"다음을 마스킹해줘: {sample}")]})
print("원문:", sample)
print("결과:", msg_text(result["messages"][-1].content)[:180])

print("\n✅ ch02 개인정보 마스킹 에이전트 실습 완료")
