"""
랭체인과 랭그래프로 AI 에이전트 개발하기 — LangChain 에이전트 실습
(원본: LangChain/랭체인에서_에이전트_사용하기.ipynb)

【교재→2026 업데이트】
  - from langchain.chat_models import ChatOpenAI
    → from langchain_google_genai import ChatGoogleGenerativeAI
  - initialize_agent() + AgentType → 완전 제거
    → create_react_agent (langgraph.prebuilt) 사용
  - load_tools(["wikipedia", "llm-math"]) → 커스텀 도구로 대체
    (wikipedia 패키지 의존성 제거 → 경량 환경 유지)
  - from langchain.memory import ConversationBufferMemory → 직접 관리
  - AgentExecutor → 제거, create_react_agent 표준 패턴 사용
  - DocstoreExplorer(Wikipedia()) → 커스텀 search 도구로 대체
"""
import os, math
from dotenv import load_dotenv
load_dotenv()

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent

llm = ChatGoogleGenerativeAI(
    model=os.getenv("GEMINI_MODEL", "gemini-2.5-flash"),
    temperature=0,
)

def msg_text(content) -> str:
    """Gemini 2.5는 응답을 list[dict] (thought signature 포함) 형태로 줄 때가 있어
    문자열로 평탄화한다. (ch03과 동일 가드)"""
    if isinstance(content, list):
        return " ".join(
            c.get("text", "") for c in content if isinstance(c, dict)
        ).strip()
    return str(content)

# ── 1. 기초 도구 정의 (교재의 wikipedia + llm-math 대체) ──────────────────
@tool
def wikipedia_search(query: str) -> str:
    """주제에 대한 간단한 설명을 반환합니다. (실습용 더미 — 실제 사용 시 wikipedia 패키지 연결)"""
    knowledge = {
        "소나무": "소나무는 침엽수로, 봄(3~4월)이나 가을(9~10월)이 이식에 가장 적합한 계절입니다.",
        "인공지능": "인공지능(AI)은 컴퓨터 시스템이 인간처럼 학습하고 문제를 해결하는 기술입니다.",
        "랭체인": "LangChain은 LLM 기반 애플리케이션 개발을 위한 오픈소스 프레임워크입니다.",
        "밍크선인장": "밍크 선인장은 건조한 환경을 선호하며, 과습을 피하고 통풍이 잘 되는 곳에 두어야 합니다.",
    }
    for key, val in knowledge.items():
        if key in query:
            return val
    # 실제 환경에서는 wikipedia 패키지 사용: import wikipedia; return wikipedia.summary(query, sentences=2)
    return f"'{query}'에 대한 정보: LLM이 직접 답변합니다."

@tool
def calculator(expression: str) -> str:
    """수학 계산을 수행합니다. 파이썬 수식을 입력받습니다. 예: '2+2', 'math.sqrt(16)'"""
    try:
        result = eval(expression, {"math": math, "__builtins__": {}})
        return f"계산 결과: {result}"
    except Exception as e:
        return f"계산 오류: {e}"

# ── 2. 기본 ReAct 에이전트 (교재 Zero-shot ReAct 대응) ──────────────────────
print("=== 1. Zero-shot ReAct 에이전트 ===")
agent = create_react_agent(
    llm,
    [wikipedia_search, calculator],
    prompt="도구를 사용해 사용자 질문에 답변하세요.",
)

queries = [
    "소나무 옮겨심기 좋은 계절은?",
    "밍크 선인장 키우는 방법은?",
    "피타고라스 정리에서 빗변이 5, 밑변이 3이면 높이는? math.sqrt(5**2 - 3**2)로 계산해",
]

for query in queries:
    result = agent.invoke({"messages": [("human", query)]})
    print(f"\nQ: {query}")
    print(f"A: {msg_text(result['messages'][-1].content)[:150]}")

# ── 3. 대화형 에이전트 (교재 Conversational ReAct 대응) ──────────────────────
print("\n=== 2. 대화형 에이전트 (메모리 포함) ===")
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

class ConversationalAgent:
    """교재의 ConversationBufferMemory + initialize_agent 패턴을 LangChain 1.x 방식으로 대체"""
    def __init__(self, tools, max_turns=3):
        self.agent = create_react_agent(
            llm, tools,
            prompt="당신은 대화 내용을 기억하는 친절한 AI 어시스턴트입니다.",
        )
        self.history = []
        self.max_turns = max_turns  # 교재의 max_iterations 대응

    def chat(self, user_input: str) -> str:
        self.history.append(("human", user_input))
        # 최근 max_turns 번만 유지
        windowed = self.history[-self.max_turns * 2:]
        result = self.agent.invoke({"messages": windowed})
        answer = msg_text(result["messages"][-1].content)
        self.history.append(("assistant", answer))
        return answer

conv_agent = ConversationalAgent([wikipedia_search, calculator], max_turns=3)
print(f"사용자: AI 에이전트란 뭐야?")
print(f"봇: {conv_agent.chat('AI 에이전트란 뭐야?')[:120]}...")
print(f"\n사용자: 랭체인이랑 어떤 관계야?")
print(f"봇: {conv_agent.chat('랭체인이랑 어떤 관계야?')[:120]}...")

print("\n✅ ch01 LangChain 에이전트 실습 완료")
