"""
랭체인과 랭그래프로 AI 에이전트 개발하기 — LangGraph 멀티 에이전트
(원본: LangGraph/멀티_에이전트_생성하기.ipynb)

【교재→2026 업데이트】
  - from langchain_openai import ChatOpenAI → ChatGoogleGenerativeAI
  - from langchain.prompts import PromptTemplate → langchain_core.prompts
  - from langchain.chains import RetrievalQA → LCEL 기반 체인
  - from langchain.llms import OpenAI → 제거 (ChatGoogleGenerativeAI 통일)
  - LangGraph API: StateGraph, add_node, add_conditional_edges → 동일 (하위 호환)
  - ✅ LangGraph 1.2.x 검증 완료
"""
import os
from dotenv import load_dotenv
load_dotenv()

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import StateGraph, END
from typing import TypedDict, Literal

llm = ChatGoogleGenerativeAI(
    model=os.getenv("GEMINI_MODEL", "gemini-2.5-flash"),
    temperature=0,
)

# ── 상태 정의 ──────────────────────────────────────────────────────────────
class AgentState(TypedDict):
    input: str
    output: str
    decision: str  # "코드 질문" 또는 "일반 질문"

# ── 노드 함수 ──────────────────────────────────────────────────────────────
def analyze_question(state: AgentState) -> dict:
    """질문 유형을 분류하는 노드 (교재 analyze_question 함수 대응)"""
    prompt = PromptTemplate.from_template("""
다음 질문이 '코드 질문'인지 '일반 질문'인지 분류하세요.
'코드 질문' 또는 '일반 질문' 중 하나만 답하세요.

질문: {input}
분류:""")
    chain = prompt | llm | StrOutputParser()
    decision = chain.invoke({"input": state["input"]}).strip()
    # 정규화
    if "코드" in decision:
        decision = "코드 질문"
    else:
        decision = "일반 질문"
    return {"decision": decision}

def code_question(state: AgentState) -> dict:
    """코드 질문 처리 노드"""
    prompt = PromptTemplate.from_template(
        "파이썬 개발자로서 다음 질문에 코드 예시와 함께 답변하세요:\n{input}"
    )
    chain = prompt | llm | StrOutputParser()
    answer = chain.invoke({"input": state["input"]})
    return {"output": f"[코드 답변]\n{answer}"}

def general_question(state: AgentState) -> dict:
    """일반 질문 처리 노드"""
    prompt = PromptTemplate.from_template(
        "친절하고 명확하게 다음 질문에 답변하세요:\n{input}"
    )
    chain = prompt | llm | StrOutputParser()
    answer = chain.invoke({"input": state["input"]})
    return {"output": f"[일반 답변]\n{answer}"}

def route_decision(state: AgentState) -> Literal["code_agent", "general_agent"]:
    """분기 결정 함수"""
    if state["decision"] == "코드 질문":
        return "code_agent"
    return "general_agent"

# ── 그래프 구성 ────────────────────────────────────────────────────────────
workflow = StateGraph(AgentState)
workflow.add_node("analyze", analyze_question)
workflow.add_node("code_agent", code_question)
workflow.add_node("general_agent", general_question)

workflow.set_entry_point("analyze")
workflow.add_conditional_edges("analyze", route_decision, {
    "code_agent": "code_agent",
    "general_agent": "general_agent",
})
workflow.add_edge("code_agent", END)
workflow.add_edge("general_agent", END)

graph = workflow.compile()

# ── 실행 테스트 ────────────────────────────────────────────────────────────
print("=== LangGraph 멀티 에이전트 (질문 유형 분기) ===\n")

test_cases = [
    "프롬프트란?",  # 일반 질문
    "숫자 입력을 받아 짝수/홀수를 판별하는 파이썬 코드를 작성해줘",  # 코드 질문
    "LangGraph가 뭔지 설명해줘",  # 일반 질문
]

for query in test_cases:
    result = graph.invoke({"input": query, "output": "", "decision": ""})
    print(f"Q: {query}")
    print(f"분류: {result['decision']}")
    print(f"A: {result['output'][:150]}...")
    print("-" * 60)

print("\n✅ ch02 LangGraph 멀티 에이전트 실습 완료")
