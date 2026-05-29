"""
두잇! LLM 활용 에이전트 입문 — 4장 실습
: 메모리·대화형 챗봇

【교재→2026 업데이트】
  - ConversationBufferMemory → 메시지 리스트 직접 관리
    교재: memory = ConversationBufferMemory()
          chain = ConversationChain(llm=llm, memory=memory)
    현재: messages = []로 직접 관리 (LangChain 1.x 권장 방식)
"""
import os
from dotenv import load_dotenv
load_dotenv()

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

llm = ChatGoogleGenerativeAI(model=os.getenv("GEMINI_MODEL", "gemini-2.5-flash"), temperature=0.7)

class SimpleChatbot:
    def __init__(self, system_prompt: str):
        self.messages = [SystemMessage(content=system_prompt)]
        self.llm = llm

    def chat(self, user_input: str) -> str:
        self.messages.append(HumanMessage(content=user_input))
        response = self.llm.invoke(self.messages)
        self.messages.append(AIMessage(content=response.content))
        return response.content

bot = SimpleChatbot("당신은 친절한 여행 가이드입니다.")
print(bot.chat("일본 여행을 계획 중이에요."))
print(bot.chat("도쿄에서 꼭 가볼 곳 3군데 추천해줘."))
print(bot.chat("방금 추천한 첫 번째 장소에 대해 더 알려줘."))  # 맥락 기억 테스트

# 슬라이딩 윈도우 메모리
class WindowChatbot:
    def __init__(self, system_prompt: str, window_size: int = 6):
        self.system = SystemMessage(content=system_prompt)
        self.history = []
        self.window_size = window_size

    def chat(self, user_input: str) -> str:
        self.history.append(HumanMessage(content=user_input))
        windowed = self.history[-self.window_size:]
        response = llm.invoke([self.system] + windowed)
        self.history.append(AIMessage(content=response.content))
        return response.content

print("\n=== 슬라이딩 윈도우 챗봇 ===")
wbot = WindowChatbot("당신은 간결하게 답하는 비서입니다.", window_size=4)
print(wbot.chat("내 이름은 수미야."))
print(wbot.chat("내 이름이 뭐라고 했지?"))

print("\n✅ ch04 메모리·대화형 챗봇 실습 완료")
