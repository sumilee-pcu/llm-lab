"""
두잇! LLM 활용 에이전트 입문 — 6장 실습
: LangGraph 멀티에이전트 워크플로

【교재→2026 업데이트】
  - LangGraph 1.2.x 에서 검증 완료 (하위 호환 유지)
  - StateGraph 타입 힌트 강화 권장
  - ✅ add_conditional_edges 시그니처 변경 없음
"""
import os
from dotenv import load_dotenv
load_dotenv()

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, END
from typing import TypedDict, Literal

llm = ChatGoogleGenerativeAI(model=os.getenv("GEMINI_MODEL", "gemini-2.5-flash"), temperature=0)

# 1. 기본 StateGraph (답변 → 번역)
class SimpleState(TypedDict):
    question: str
    answer: str
    language: str

def answer_node(state):
    response = llm.invoke([HumanMessage(content=state["question"])])
    return {"answer": response.content}

def translate_node(state):
    prompt = f"다음 텍스트를 {state['language']}로 번역해줘:\n{state['answer']}"
    response = llm.invoke([HumanMessage(content=prompt)])
    return {"answer": response.content}

graph = StateGraph(SimpleState)
graph.add_node("answer", answer_node)
graph.add_node("translate", translate_node)
graph.set_entry_point("answer")
graph.add_edge("answer", "translate")
graph.add_edge("translate", END)
app = graph.compile()
result = app.invoke({"question": "파이썬이란?", "answer": "", "language": "영어"})
print(result["answer"])

# 2. 조건부 분기 라우터 패턴
class RouterState(TypedDict):
    query: str
    category: str
    response: str

def router_node(state):
    prompt = f"""다음 질문의 카테고리를 한 단어로만 답해줘 (선택지: 수학, 언어, 기타):\n질문: {state['query']}"""
    response = llm.invoke([HumanMessage(content=prompt)])
    cat = response.content.strip()
    if "수학" in cat: return {"category": "math"}
    elif "언어" in cat or "번역" in cat: return {"category": "language"}
    else: return {"category": "general"}

def math_node(state):
    resp = llm.invoke([HumanMessage(content=f"수학 전문가로서: {state['query']}")])
    return {"response": f"[수학] {resp.content[:100]}"}

def language_node(state):
    resp = llm.invoke([HumanMessage(content=f"언어 전문가로서: {state['query']}")])
    return {"response": f"[언어] {resp.content[:100]}"}

def general_node(state):
    resp = llm.invoke([HumanMessage(content=state["query"])])
    return {"response": f"[일반] {resp.content[:100]}"}

def route_selector(state) -> Literal["math", "language", "general"]:
    return state["category"]

router_graph = StateGraph(RouterState)
for name, fn in [("router", router_node), ("math", math_node), ("language", language_node), ("general", general_node)]:
    router_graph.add_node(name, fn)
router_graph.set_entry_point("router")
router_graph.add_conditional_edges("router", route_selector, {"math": "math", "language": "language", "general": "general"})
for node in ["math", "language", "general"]:
    router_graph.add_edge(node, END)
router_app = router_graph.compile()

for query in ["피타고라스 정리를 설명해줘", "영어로 감사합니다는?", "오늘 점심 추천"]:
    result = router_app.invoke({"query": query, "category": "", "response": ""})
    print(f"Q: {query}\nA: {result['response'][:80]}\n")

print("\n✅ ch06 LangGraph 멀티에이전트 워크플로 실습 완료")
