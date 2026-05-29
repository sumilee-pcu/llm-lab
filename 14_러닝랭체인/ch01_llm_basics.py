"""
러닝 랭체인 — 1장 LLM 기초 & LCEL
(원본: ch1/py/*.py — a-llm.py ~ l-declarative.py)

【교재→2026 업데이트】
  - from langchain_openai import ChatOpenAI → ChatGoogleGenerativeAI
  - ChatOpenAI(model="gpt-3.5-turbo") → ChatGoogleGenerativeAI(model="gemini-2.5-flash")
  - langchain==0.2.x → 1.x (동일 LCEL 패턴, 하위 호환)
  - OpenAI structured output → with_structured_output() (Gemini도 지원)
  - streaming → .stream() 메서드 동일
  - ✅ LangChain 1.x + Gemini 검증 완료
"""
import os
from dotenv import load_dotenv
load_dotenv()

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import SystemMessage, HumanMessage
from pydantic import BaseModel, Field

GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
model = ChatGoogleGenerativeAI(model=GEMINI_MODEL, temperature=0)

# ── a. 기본 LLM 호출 ──────────────────────────────────────────────────────
print("=== a. 기본 LLM 호출 ===")
response = model.invoke("하늘은")
print(response.content[:100])

# ── b. 채팅 모델 (메시지 리스트) ──────────────────────────────────────────
print("\n=== b. 채팅 모델 ===")
messages = [
    SystemMessage(content="당신은 친절한 AI 어시스턴트입니다."),
    HumanMessage(content="LangChain이란 무엇인가요?"),
]
response = model.invoke(messages)
print(response.content[:150])

# ── c. 시스템 메시지 역할 부여 ────────────────────────────────────────────
print("\n=== c. 시스템 역할 ===")
messages = [
    ("system", "당신은 파이썬 전문가입니다. 간결하게 답하세요."),
    ("human", "리스트 컴프리헨션이란?"),
]
print(model.invoke(messages).content[:150])

# ── d. 프롬프트 템플릿 ────────────────────────────────────────────────────
print("\n=== d. 프롬프트 템플릿 ===")
prompt = ChatPromptTemplate.from_messages([
    ("system", "당신은 {language} 전문가입니다."),
    ("human", "{question}"),
])
formatted = prompt.invoke({"language": "파이썬", "question": "제너레이터란?"})
print(model.invoke(formatted).content[:150])

# ── l. LCEL 선언적 체인 ──────────────────────────────────────────────────
print("\n=== l. LCEL 선언적 체인 ===")
chatbot = ChatPromptTemplate.from_messages([
    ("system", "당신은 친절한 AI 어시스턴트입니다."),
    ("human", "{question}"),
]) | model | StrOutputParser()

response = chatbot.invoke({"question": "어떤 LLM 제공자가 있나요?"})
print(response[:150])

# ── ka. 스트리밍 ─────────────────────────────────────────────────────────
print("\n=== ka. 스트리밍 ===")
print("스트리밍 출력: ", end="", flush=True)
for chunk in chatbot.stream({"question": "LangChain의 주요 기능 3가지를 간단히 나열하세요."}):
    print(chunk, end="", flush=True)
print()

# ── h. 구조화된 출력 (structured output) ──────────────────────────────────
print("\n=== h. 구조화된 출력 ===")
class BookInfo(BaseModel):
    title: str = Field(description="책 제목")
    author: str = Field(description="저자 이름")
    year: int = Field(description="출판 연도")

structured_model = model.with_structured_output(BookInfo)
result = structured_model.invoke("러닝 랭체인은 LangChain 창업자가 쓴 책으로 2024년 오라일리에서 출판되었습니다.")
print(f"제목: {result.title}, 저자: {result.author}, 연도: {result.year}")

print("\n✅ ch01 LLM 기초 & LCEL 실습 완료")
