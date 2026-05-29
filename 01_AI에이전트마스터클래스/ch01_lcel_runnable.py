"""
AI 에이전트 마스터 클래스 (한빛) — 1부: LCEL & Runnable
강의 요소: LCEL 데이터 처리 체인 / Runnable 병렬·조건부 로직

【교재→2026 업데이트】
  - ChatOpenAI → ChatGoogleGenerativeAI (gemini-2.5-flash)
  - LLMChain → LCEL (prompt | llm | parser)
  - RunnableParallel / RunnableBranch 로 병렬·조건부 (LangChain 1.x)
"""
import os
from dotenv import load_dotenv
load_dotenv()

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnableBranch, RunnableLambda

llm = ChatGoogleGenerativeAI(model=os.getenv("GEMINI_MODEL", "gemini-2.5-flash"), temperature=0)

# 1. 기본 LCEL 체인
print("=== 1. LCEL 기본 체인 ===")
chain = ChatPromptTemplate.from_template("{wine}와 어울리는 음식 1가지만 짧게.") | llm | StrOutputParser()
print(chain.invoke({"wine": "까베르네 소비뇽"}).strip()[:80])

# 2. RunnableParallel — 한 입력으로 여러 결과 병렬 생성
print("\n=== 2. RunnableParallel (병렬) ===")
food = ChatPromptTemplate.from_template("{wine}와 어울리는 음식 1개만.") | llm | StrOutputParser()
mood = ChatPromptTemplate.from_template("{wine}의 분위기를 한 단어로.") | llm | StrOutputParser()
parallel = RunnableParallel(food=food, mood=mood)
res = parallel.invoke({"wine": "샴페인"})
print("food:", res["food"].strip()[:50])
print("mood:", res["mood"].strip()[:30])

# 3. RunnableBranch — 조건부 라우팅
print("\n=== 3. RunnableBranch (조건부) ===")
red = ChatPromptTemplate.from_template("레드와인 {q} 추천 한 줄.") | llm | StrOutputParser()
white = ChatPromptTemplate.from_template("화이트와인 {q} 추천 한 줄.") | llm | StrOutputParser()
default = ChatPromptTemplate.from_template("와인 일반 추천: {q} 한 줄.") | llm | StrOutputParser()
branch = RunnableBranch(
    (lambda x: "레드" in x["q"], red),
    (lambda x: "화이트" in x["q"], white),
    default,
)
for q in ["레드 스테이크용", "화이트 해산물용", "디저트용"]:
    print(f"[{q}] {branch.invoke({'q': q}).strip()[:50]}")

print("\n✅ ch01 LCEL & Runnable 실습 완료")
