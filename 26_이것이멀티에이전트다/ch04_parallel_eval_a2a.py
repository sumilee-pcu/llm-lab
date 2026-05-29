"""
이것이 멀티 에이전트다 (한빛) — 멀티: 병렬 검색 + 반복 평가 품질 개선(A2A)
강의 요소: 병렬 검색·반복 평가 품질개선 (A2A 오케스트레이션)

【교재→2026 업데이트】
  - 여러 관점 답변을 병렬 생성 → 평가 에이전트가 최고안 선택 → 부족하면 1회 보강
  - LangGraph 없이도 개념이 명확하도록 인프로세스 오케스트레이션으로 구성
"""
import os, re
from dotenv import load_dotenv
load_dotenv()

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

llm = ChatGoogleGenerativeAI(model=os.getenv("GEMINI_MODEL", "gemini-2.5-flash"), temperature=0.4)
parse = StrOutputParser()

# 서로 다른 관점의 워커(병렬 검색/생성 대체)
perspectives = {
    "기술관점": "기술적 관점에서 한 문장으로 답하세요: {q}",
    "비용관점": "비용/효율 관점에서 한 문장으로 답하세요: {q}",
}
judge = ChatPromptTemplate.from_template(
    "질문 '{q}'에 대한 후보 답변들:\n{cands}\n"
    "가장 우수한 답변 번호와 1~5점을 'BEST: n SCORE: s'로만 출력."
) | llm | parse

question = "사내 문서 검색에 RAG를 도입할 때 핵심 고려사항은?"

# 1) 병렬(순차 호출이지만 독립) 워커 생성 — A2A의 멀티 워커
candidates = []
for name, tmpl in perspectives.items():
    ans = (ChatPromptTemplate.from_template(tmpl) | llm | parse).invoke({"q": question}).strip()
    candidates.append(f"{name}: {ans}")
    print(f"[{name}] {ans[:80]}")

# 2) 평가 에이전트(A2A) — 최고안·점수
verdict = judge.invoke({"q": question, "cands": "\n".join(f"{i+1}) {c}" for i, c in enumerate(candidates))})
m = re.search(r"BEST:\s*(\d+).*SCORE:\s*([1-5])", verdict)
print(f"\n평가 결과: {verdict.strip()[:60]}")

# 3) 반복 개선: 점수 낮으면 보강 1회
if m and int(m.group(2)) < 4:
    print("→ 점수 낮음: 보강 라운드 실행")
    best = candidates[int(m.group(1)) - 1]
    improved = (ChatPromptTemplate.from_template("다음 답변을 더 구체적으로 보강: {a}") | llm | parse).invoke({"a": best})
    print("보강안:", improved.strip()[:100])
else:
    print("→ 품질 충족: 최종 채택")

print("\n✅ ch04 병렬검색·반복평가(A2A) 실습 완료")
