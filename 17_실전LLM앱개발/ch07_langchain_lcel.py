"""
실전 LLM 앱 개발 — 7장 LangChain LCEL 실습
(원본: 7_langchain/LCEL.ipynb, langchain_prompt_template.ipynb, langchain_runnable_workflow.ipynb)

【교재→2026 업데이트】
  - from langchain_openai import ChatOpenAI → ChatGoogleGenerativeAI
  - langchain==0.2.16 → langchain 1.x (LLMChain 완전 제거됨)
    구 방식: LLMChain(llm=llm, prompt=prompt).run(...)
    신 방식: LCEL (prompt | llm | parser).invoke(...)
  - RunnablePassthrough, RunnableParallel, RunnableLambda → 동일 (하위 호환)
  - from google.colab import userdata → python-dotenv
  - ✅ LangChain 1.x + Gemini 검증 완료
"""
import os
from dotenv import load_dotenv
load_dotenv()

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableParallel, RunnableLambda

GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
llm = ChatGoogleGenerativeAI(model=GEMINI_MODEL, temperature=0.7)

# ── 1. 기본 LCEL 체인 ─────────────────────────────────────────────────────
print("=== 1. 기본 LCEL 체인 ===")
prompt1 = ChatPromptTemplate.from_template("{word}을 주제로 짧은 시를 써줘.")
chain1 = prompt1 | llm | StrOutputParser()
result1 = chain1.invoke({"word": "봄"})
print(result1[:200])

# ── 2. 프롬프트 템플릿 활용 (ChatPromptTemplate) ──────────────────────────
print("\n=== 2. 프롬프트 템플릿 (시스템 + 사용자) ===")
prompt2 = ChatPromptTemplate.from_messages([
    ("system", "당신은 {language} 전문가 개발자입니다. 간결하게 답변하세요."),
    ("human", "{question}"),
])
chain2 = prompt2 | llm | StrOutputParser()
result2 = chain2.invoke({"language": "파이썬", "question": "리스트 컴프리헨션이란?"})
print(result2[:200])

# ── 3. RunnablePassthrough — 입력 그대로 전달 ─────────────────────────────
print("\n=== 3. RunnablePassthrough ===")
prompt3 = ChatPromptTemplate.from_template("{topic}을 한 문장으로 설명하세요.")
chain3 = {"topic": RunnablePassthrough()} | prompt3 | llm | StrOutputParser()
result3 = chain3.invoke("머신러닝")
print(result3[:200])

# ── 4. RunnableParallel — 병렬 처리 ─────────────────────────────────────
print("\n=== 4. RunnableParallel (병렬 처리) ===")
llm_precise = ChatGoogleGenerativeAI(model=GEMINI_MODEL, temperature=0)

pros_chain = (
    ChatPromptTemplate.from_template("{topic}의 장점 2가지를 간결하게 나열하세요.")
    | llm_precise | StrOutputParser()
)
cons_chain = (
    ChatPromptTemplate.from_template("{topic}의 단점 2가지를 간결하게 나열하세요.")
    | llm_precise | StrOutputParser()
)

parallel_chain = RunnableParallel(
    pros=pros_chain,
    cons=cons_chain,
)
result4 = parallel_chain.invoke({"topic": "LLM API 사용"})
print(f"장점: {result4['pros'][:120]}")
print(f"단점: {result4['cons'][:120]}")

# ── 5. 체인 연결 (Chain of Chains) ───────────────────────────────────────
print("\n=== 5. 체인 연결 ===")
first_prompt = ChatPromptTemplate.from_template("{topic}의 핵심 개념을 한 줄로 요약하세요.")
second_prompt = ChatPromptTemplate.from_template(
    "다음 개념을 초등학생이 이해할 수 있도록 쉽게 설명하세요: {summary}"
)

chain_of_chains = (
    {"summary": first_prompt | llm_precise | StrOutputParser()}
    | second_prompt | llm_precise | StrOutputParser()
)
result5 = chain_of_chains.invoke({"topic": "벡터 임베딩"})
print(result5[:200])

print("\n✅ ch07 LangChain LCEL 실습 완료")
