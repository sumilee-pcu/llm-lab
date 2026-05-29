"""
요즘 AI 에이전트 개발 — A2A(Agent-to-Agent) 통신 패턴
강의 요소: A2A 에이전트 통신 / (MCP는 21번 컨텍스트 엔지니어링 폴더 실습 참조)

【교재→2026 업데이트】
  - A2A 프로토콜의 핵심 개념(요청-응답 메시지 교환)을 두 Gemini 에이전트로 시연
  - 실제 A2A 서버/네트워크 없이 인프로세스 메시지 패싱으로 개념 실습
"""
import os
from dotenv import load_dotenv
load_dotenv()

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage

llm = ChatGoogleGenerativeAI(model=os.getenv("GEMINI_MODEL", "gemini-2.5-flash"), temperature=0)


class Agent:
    """A2A의 에이전트 — system 역할을 갖고 메시지를 받아 응답"""
    def __init__(self, name, role):
        self.name, self.role = name, role

    def handle(self, message: str) -> str:
        resp = llm.invoke([SystemMessage(content=self.role), HumanMessage(content=message)])
        return resp.content.strip()


# 두 전문 에이전트: 질문 에이전트 ↔ 전문가 에이전트 (A2A 교환)
researcher = Agent("연구원", "당신은 질문을 한 문장으로 더 구체화하는 연구원입니다.")
expert = Agent("전문가", "당신은 질문에 2문장 이내로 답하는 도메인 전문가입니다.")

user_q = "RAG 성능을 높이려면?"
print(f"[사용자] {user_q}")
refined = researcher.handle(f"이 질문을 더 구체적으로 한 문장으로: {user_q}")
print(f"[연구원→전문가] {refined[:90]}")
answer = expert.handle(refined)
print(f"[전문가→사용자] {answer[:140]}")

print("\n✅ ch04 A2A 통신 패턴 실습 완료")
