"""
LLM 개발 애플리케이션 — LLMOps 맛보기: LLM-as-judge 평가
강의 요소: MLOps/LLMOps (품질 평가·운영)

【교재→2026 업데이트】
  - 평가 자동화: LLM이 답변 품질을 점수화(LLM-as-judge) → 운영 지표화
  - 모든 호출 Gemini (gemini-2.5-flash)
"""
import os, re
from dotenv import load_dotenv
load_dotenv()

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

llm = ChatGoogleGenerativeAI(model=os.getenv("GEMINI_MODEL", "gemini-2.5-flash"), temperature=0)

answer_chain = ChatPromptTemplate.from_template("질문에 2문장 이내로 답하세요: {q}") | llm | StrOutputParser()
judge_chain = ChatPromptTemplate.from_template(
    "다음 답변의 정확성과 명료성을 1~5점으로 평가하고 'SCORE: N' 형식으로만 끝에 표기하세요.\n"
    "[질문] {q}\n[답변] {a}"
) | llm | StrOutputParser()

questions = ["LLM의 환각(hallucination)이란?", "파인튜닝과 RAG의 차이는?"]
scores = []
for q in questions:
    a = answer_chain.invoke({"q": q}).strip()
    verdict = judge_chain.invoke({"q": q, "a": a})
    m = re.search(r"SCORE:\s*([1-5])", verdict)
    score = int(m.group(1)) if m else None
    scores.append(score)
    print(f"Q: {q}\n A: {a[:80]}\n 평가점수: {score}\n")

valid = [s for s in scores if s]
if valid:
    print(f"평균 품질 점수: {sum(valid)/len(valid):.1f} / 5  (운영 모니터링 지표 예시)")

print("\n✅ ch03 LLM-as-judge 평가 실습 완료")
