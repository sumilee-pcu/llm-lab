"""
랭체인과 랭그래프로 AI 에이전트 개발하기 — LangGraph ReAct 에이전트
(원본: LangGraph/LangGraph에서_ReAct_에이전트_생성하기.ipynb)

【교재→2026 업데이트】
  - ChatOpenAI(model="gpt-4o") → ChatGoogleGenerativeAI(model="gemini-2.5-flash")
  - LangGraph MessagesState 사용 패턴 — 1.2.x 기준 동일
  - ToolNode 임포트: langgraph.prebuilt → 동일 (하위 호환 유지)
  - streaming=True 파라미터: Gemini에서도 지원 (.stream() 메서드로)
  - ✅ LangGraph 1.2.x 검증 완료
"""
import os, math
from dotenv import load_dotenv
load_dotenv()

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent, ToolNode
from langgraph.graph import StateGraph, MessagesState, END

llm = ChatGoogleGenerativeAI(
    model=os.getenv("GEMINI_MODEL", "gemini-2.5-flash"),
    temperature=0,
)

# ── 도구 정의 ──────────────────────────────────────────────────────────────
@tool
def search(query: str) -> str:
    """웹 검색을 수행합니다. (실습용 더미 데이터)"""
    results = {
        "LangGraph": "LangGraph는 LLM 기반 에이전트/멀티에이전트 워크플로를 위한 라이브러리로, LangChain 위에서 동작합니다.",
        "AI 에이전트": "AI 에이전트는 환경을 인식하고, 목표를 설정하며, 도구를 사용해 자율적으로 작업을 수행하는 시스템입니다.",
        "RAG": "RAG(Retrieval-Augmented Generation)는 외부 지식을 검색해 LLM 응답의 정확도를 높이는 기법입니다.",
    }
    for key, val in results.items():
        if key.lower() in query.lower():
            return val
    return f"'{query}' 검색 결과: 관련 정보를 LLM이 직접 생성합니다."

@tool
def calculate(expression: str) -> str:
    """수학 계산을 수행합니다."""
    try:
        result = eval(expression, {"math": math, "__builtins__": {}})
        return f"{expression} = {result}"
    except Exception as e:
        return f"오류: {e}"

tools = [search, calculate]

# ── 1. create_react_agent 방식 (간단, 권장) ──────────────────────────────
print("=== 1. create_react_agent (권장 패턴) ===")
agent = create_react_agent(llm, tools)

result = agent.invoke({
    "messages": [("human", "LangGraph가 무엇인지 검색해주고, 3의 제곱도 계산해줘")]
})
content = result["messages"][-1].content
if isinstance(content, list):
    content = " ".join(c.get("text", "") for c in content if isinstance(c, dict))
print(content[:200])

# ── 2. 수동 ReAct 그래프 구성 (교재 패턴 — 내부 동작 이해용) ─────────────
print("\n=== 2. 수동 ReAct 그래프 (교재 패턴 이해) ===")

llm_with_tools = llm.bind_tools(tools)
tool_node = ToolNode(tools)

def call_model(state: MessagesState) -> dict:
    messages = state["messages"]
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}

def should_continue(state: MessagesState) -> str:
    last_message = state["messages"][-1]
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"
    return END

manual_graph = StateGraph(MessagesState)
manual_graph.add_node("agent", call_model)
manual_graph.add_node("tools", tool_node)
manual_graph.set_entry_point("agent")
manual_graph.add_conditional_edges("agent", should_continue, {"tools": "tools", END: END})
manual_graph.add_edge("tools", "agent")

manual_app = manual_graph.compile()

result2 = manual_app.invoke({
    "messages": [("human", "AI 에이전트가 뭔지 검색해줘")]
})
content2 = result2["messages"][-1].content
if isinstance(content2, list):
    content2 = " ".join(c.get("text", "") for c in content2 if isinstance(c, dict))
print(content2[:200])

print("\n✅ ch03 LangGraph ReAct 에이전트 실습 완료")
