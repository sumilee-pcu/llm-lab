"""
두잇! LLM 활용 에이전트 입문 — 2장 실습
: API 첫 호출 & 프롬프트 기초

【교재→2026 업데이트】
  - OpenAI API → Gemini API (google-genai 2.x)
  - openai.ChatCompletion.create() → ChatGoogleGenerativeAI.invoke()
  - 모델명: gpt-3.5-turbo/gpt-4 → gemini-2.5-flash
  - API 키 환경변수: OPENAI_API_KEY → GOOGLE_API_KEY
"""
import os
from dotenv import load_dotenv
load_dotenv()

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage

llm = ChatGoogleGenerativeAI(
    model=os.getenv("GEMINI_MODEL", "gemini-2.5-flash"),
    temperature=0.7,
)

# 1. 기본 호출
response = llm.invoke([HumanMessage(content="파이썬으로 Hello World를 출력하는 코드를 알려줘.")])
print(response.content)

# 2. 시스템 프롬프트 활용
messages = [
    SystemMessage(content="당신은 초등학생에게 설명하는 친절한 선생님입니다."),
    HumanMessage(content="인공지능이 뭐예요?"),
]
response = llm.invoke(messages)
print(response.content)

# 3. 스트리밍 응답
for chunk in llm.stream([HumanMessage(content="LLM이 무엇인지 두 문장으로 설명해줘.")]):
    print(chunk.content, end="", flush=True)
print()

print("\n✅ ch02 API 첫 호출 & 프롬프트 기초 실습 완료")
