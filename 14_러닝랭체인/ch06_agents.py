"""
러닝 랭체인 — 6장 에이전트
(원본: ch6/py/*.py — a-basic-agent.py, b-force-first-tool.py, c-many-tools.py)

【교재→2026 업데이트】
  - ChatOpenAI → ChatGoogleGenerativeAI
  - DuckDuckGoSearchRun → 더미 search 도구 (외부 API 의존성 제거, 경량 환경)
  - StateGraph + ToolNode + tools_condition 패턴 → 동일 (LangGraph 1.2.x 하위 호환)
  - START 노드 명시적 사용 → 동일
  - ✅ LangGraph 1.2.x + Gemini 검증 완료
"""
import os, ast, math
from dotenv import load_dotenv
load_dotenv()

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from typing import Annotated, TypedDict

GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

# ── 도구 정의 ─────────────────────────────────────────────────────────────
@tool
def calculator(query: str) -> str:
    """수학 계산 도구. 파이썬 수식을 입력받아 계산합니다. 예: '2+2', 'math.sqrt(16)'"""
    try:
        result = eval(query, {"math": math, "__builtins__": {}})
        return str(result)
    except Exception as e:
        return f"계산 오류: {e}"

@tool
def search(query: str) -> str:
    """웹 검색 도구. 주어진 질의에 대한 정보를 반환합니다. (실습용 더미)"""
    knowledge = {
        "coolidge": "Calvin Coolidge (1872~1933)는 미국 30대 대통령으로, 사망 시 60세였습니다.",
        "langchain": "LangChain은 Harrison Chase가 2022년 설립한 LLM 프레임워크 회사입니다.",
        "gemini": "Gemini는 Google DeepMind가 개발한 멀티모달 AI 모델 시리즈입니다.",
        "python": "Python은 Guido van Rossum이 1991년 만든 범용 프로그래밍 언어입니다.",
    }
    for key, val in knowledge.items():
        if key.lower() in query.lower():
            return val
    return f"'{query}'에 대한 검색 결과: LLM이 직접 생성합니다."

# ── a. 기본 에이전트 ─────────────────────────────────────────────────────
print("=== a. 기본 에이전트 ===")
tools = [search, calculator]
model = ChatGoogleGenerativeAI(model=GEMINI_MODEL, temperature=0.1).bind_tools(tools)

class State(TypedDict):
    messages: Annotated[list, add_messages]

def model_node(state: State) -> State:
    res = model.invoke(state["messages"])
    return {"messages": res}

builder = StateGraph(State)
builder.add_node("model", model_node)
builder.add_node("tools", ToolNode(tools))
builder.add_edge(START, "model")
builder.add_conditional_edges("model", tools_condition)
builder.add_edge("tools", "model")
graph = builder.compile()

input_msg = {"messages": [HumanMessage("미국 30대 대통령은 사망 시 몇 살이었나요?")]}
final_result = graph.invoke(input_msg)
last_msg = final_result["messages"][-1]
content = last_msg.content if hasattr(last_msg, "content") else ""
if isinstance(content, list):
    content = " ".join(c.get("text","") for c in content if isinstance(c, dict))
print(content[:150])

# ── b. 첫 번째 도구 강제 사용 ────────────────────────────────────────────
print("\n=== b. 첫 번째 도구 강제 사용 ===")
from langchain_core.messages import AIMessage
import json

model_forced = ChatGoogleGenerativeAI(model=GEMINI_MODEL, temperature=0)

def model_force_search(state: State) -> State:
    """첫 번째 호출은 반드시 search 도구를 사용하도록 강제"""
    if len(state["messages"]) == 1:  # 첫 번째 사용자 메시지
        tool_call = {
            "name": "search",
            "args": {"query": state["messages"][0].content},
            "id": "forced_tool_call",
            "type": "tool_call",
        }
        return {"messages": AIMessage(content="", tool_calls=[tool_call])}
    res = model_forced.bind_tools(tools).invoke(state["messages"])
    return {"messages": res}

builder2 = StateGraph(State)
builder2.add_node("model", model_force_search)
builder2.add_node("tools", ToolNode(tools))
builder2.add_edge(START, "model")
builder2.add_conditional_edges("model", tools_condition)
builder2.add_edge("tools", "model")
graph2 = builder2.compile()

r2 = graph2.invoke({"messages": [HumanMessage("LangChain에 대해 알려줘")]})
c2 = r2["messages"][-1].content
if isinstance(c2, list):
    c2 = " ".join(c.get("text","") for c in c2 if isinstance(c, dict))
print(f"최종 답변: {c2[:120]}")

# ── c. 다중 도구 에이전트 ────────────────────────────────────────────────
print("\n=== c. 다중 도구 에이전트 ===")

@tool
def get_word_length(word: str) -> int:
    """단어의 글자 수를 반환합니다."""
    return len(word)

all_tools = [search, calculator, get_word_length]
model_multi = ChatGoogleGenerativeAI(model=GEMINI_MODEL, temperature=0).bind_tools(all_tools)

def model_multi_node(state: State) -> State:
    res = model_multi.invoke(state["messages"])
    return {"messages": res}

builder3 = StateGraph(State)
builder3.add_node("model", model_multi_node)
builder3.add_node("tools", ToolNode(all_tools))
builder3.add_edge(START, "model")
builder3.add_conditional_edges("model", tools_condition)
builder3.add_edge("tools", "model")
graph3 = builder3.compile()

r3 = graph3.invoke({"messages": [HumanMessage("2의 10제곱을 계산하고, 'LangChain'이라는 단어의 글자 수도 알려줘")]})
c3 = r3["messages"][-1].content
if isinstance(c3, list):
    c3 = " ".join(c.get("text","") for c in c3 if isinstance(c, dict))
print(f"답변: {c3[:150]}")

print("\n✅ ch06 에이전트 실습 완료")
