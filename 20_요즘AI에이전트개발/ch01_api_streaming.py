"""
요즘 AI 에이전트 개발 — LLM API 호출 & 스트리밍
강의 요소: LLM API 호출·스트리밍

【교재→2026 업데이트】
  - OpenAI API → Gemini (google-genai 2.x) + LangChain 스트리밍
  - openai.chat.completions.create(stream=True) → llm.stream(...)
"""
import os
from dotenv import load_dotenv
load_dotenv()

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage

llm = ChatGoogleGenerativeAI(model=os.getenv("GEMINI_MODEL", "gemini-2.5-flash"), temperature=0.5)

# 1. 일반 호출
print("=== 1. 일반 호출 ===")
print(llm.invoke([HumanMessage(content="AI 에이전트를 한 문장으로 정의해줘.")]).content.strip()[:100])

# 2. 스트리밍 호출
print("\n=== 2. 스트리밍 ===")
chunks = 0
for chunk in llm.stream([HumanMessage(content="에이전트 개발 3단계를 짧게 알려줘.")]):
    print(chunk.content, end="", flush=True)
    chunks += 1
print(f"\n(수신 청크 수: {chunks})")

print("\n✅ ch01 API·스트리밍 실습 완료")
