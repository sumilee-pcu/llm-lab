"""
실전 LLM 앱 개발 — 5장 Gemini API 실습
(원본: 5_gemini/learn_gemini_basics.ipynb, inference_parameters.ipynb)

【교재→2026 업데이트】
  - google.colab.userdata → python-dotenv + .env
  - google-generativeai 0.7/0.8 (구 SDK) → google-genai 2.x (신 SDK)
    구 SDK: import google.generativeai as genai
            genai.configure(api_key=...)
            genai.GenerativeModel('gemini-1.5-flash')
    신 SDK: from google import genai
            client = genai.Client(api_key=...)
            client.models.generate_content(model=..., contents=...)
  - 모델: gemini-1.5-flash → gemini-2.5-flash
  - langchain-google-genai도 함께 활용 (LangChain 1.x 호환)
  - ✅ google-genai 2.x + LangChain 1.x 검증 완료
"""
import os
from dotenv import load_dotenv
load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

# ── 신 SDK (google-genai 2.x) ─────────────────────────────────────────────
from google import genai

client = genai.Client(api_key=GOOGLE_API_KEY)

# 1. 기본 사용법 — 단일 메시지
print("=== 1. 기본 메시지 ===")
response = client.models.generate_content(
    model=GEMINI_MODEL,
    contents="인공지능에 대해 한 문장으로 설명하세요.",
)
print(response.text[:200])

# 2. 멀티턴 대화 — 채팅 세션
print("\n=== 2. 멀티턴 대화 ===")
chat = client.chats.create(model=GEMINI_MODEL)
queries = [
    "인공지능에 대해 한 문장으로 짧게 설명하세요.",
    "방금 말한 기술의 대표 활용 사례 하나만 알려주세요.",
]
for q in queries:
    print(f"[사용자]: {q}")
    resp = chat.send_message(q)
    print(f"[모델]: {resp.text[:120]}")

# 3. 추론 파라미터 (temperature, max_output_tokens)
print("\n=== 3. 추론 파라미터 ===")
from google.genai import types

creative_resp = client.models.generate_content(
    model=GEMINI_MODEL,
    contents="봄에 대한 짧은 시를 써주세요.",
    config=types.GenerateContentConfig(
        temperature=1.5,
        max_output_tokens=150,
    ),
)
print("[temperature=1.5, 창의적]")
print(creative_resp.text[:200])

precise_resp = client.models.generate_content(
    model=GEMINI_MODEL,
    contents="파이썬에서 리스트와 튜플의 차이점을 한 줄로 설명하세요.",
    config=types.GenerateContentConfig(
        temperature=0,
        max_output_tokens=100,
    ),
)
print("\n[temperature=0, 정확]")
print(precise_resp.text[:200])

# 4. LangChain 방식 (교재 langchain-google-genai 패턴)
print("\n=== 4. LangChain + Gemini ===")
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

llm = ChatGoogleGenerativeAI(model=GEMINI_MODEL, temperature=0)
prompt = ChatPromptTemplate.from_template("{topic}을 주제로 한 문장 설명을 써주세요.")
chain = prompt | llm | StrOutputParser()
result = chain.invoke({"topic": "LLM(대규모 언어 모델)"})
print(result[:200])

print("\n✅ ch05 Gemini API 실습 완료")
