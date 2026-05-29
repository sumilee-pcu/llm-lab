"""
AI 에이전트 마스터 클래스 (한빛) — Memory 기반 멀티턴 대화
강의 요소: Memory 멀티턴 대화

【교재→2026 업데이트】
  - ConversationBufferMemory / ConversationChain → 메시지 리스트 직접 관리 (LangChain 1.x 권장)
"""
import os
from dotenv import load_dotenv
load_dotenv()

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

llm = ChatGoogleGenerativeAI(model=os.getenv("GEMINI_MODEL", "gemini-2.5-flash"), temperature=0.5)


class WineSommelier:
    """대화 맥락을 기억하는 와인 추천 소믈리에 봇"""
    def __init__(self):
        self.messages = [SystemMessage(content="당신은 친절한 와인 소믈리에입니다. 짧게 답하세요.")]

    def chat(self, user_input: str) -> str:
        self.messages.append(HumanMessage(content=user_input))
        resp = llm.invoke(self.messages)
        self.messages.append(AIMessage(content=resp.content))
        return resp.content


bot = WineSommelier()
print("Q1:", "스테이크에 어울리는 와인 추천해줘.")
print("A1:", bot.chat("스테이크에 어울리는 와인 추천해줘.").strip()[:100])
print("\nQ2:", "방금 추천한 와인은 어느 나라 거야?")  # 맥락 기억 테스트
print("A2:", bot.chat("방금 추천한 와인은 어느 나라 거야?").strip()[:100])
print(f"\n대화 누적 메시지 수: {len(bot.messages)}")

print("\n✅ ch02 Memory 멀티턴 대화 실습 완료")
