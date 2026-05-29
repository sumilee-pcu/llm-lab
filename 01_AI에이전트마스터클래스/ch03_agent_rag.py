"""
AI 에이전트 마스터 클래스 (한빛) — 2부: 에이전트 + RAG
강의 요소: 에이전트(도구 선택·호출) / RAG로 내부 지식 활용

【교재→2026 업데이트】
  - initialize_agent/AgentExecutor → create_react_agent (langgraph.prebuilt)
  - OpenAIEmbeddings → GoogleGenerativeAIEmbeddings (models/gemini-embedding-2)
  - 벡터 저장소: Chroma 인메모리 (외부 의존 제거)
"""
import os
from dotenv import load_dotenv
load_dotenv()

from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent

llm = ChatGoogleGenerativeAI(model=os.getenv("GEMINI_MODEL", "gemini-2.5-flash"), temperature=0)
embeddings = GoogleGenerativeAIEmbeddings(model=os.getenv("GEMINI_EMBEDDING_MODEL", "models/gemini-embedding-2"))


def msg_text(content) -> str:
    if isinstance(content, list):
        return " ".join(c.get("text", "") for c in content if isinstance(c, dict)).strip()
    return str(content)


# ── 내부 와인 지식베이스 (RAG) ──────────────────────────────────────────────
WINE_DB = [
    "까베르네 소비뇽은 풀바디 레드와인으로 스테이크·양고기와 잘 어울립니다.",
    "샤르도네는 화이트와인으로 크림소스 파스타·해산물과 어울립니다.",
    "피노 누아는 가벼운 레드로 연어·버섯 요리와 잘 맞습니다.",
    "리슬링은 약간 달콤한 화이트로 매운 아시아 음식과 균형을 이룹니다.",
]
vectorstore = Chroma.from_documents([Document(page_content=t) for t in WINE_DB], embedding=embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 2})


@tool
def wine_knowledge(query: str) -> str:
    """내부 와인 지식베이스에서 음식 페어링 정보를 검색합니다."""
    docs = retriever.invoke(query)
    return "\n".join(d.page_content for d in docs)


@tool
def price_lookup(wine: str) -> str:
    """와인의 대략적 가격대를 반환합니다. (실습용 더미)"""
    prices = {"까베르네": "3~8만원", "샤르도네": "2~5만원", "피노": "4~10만원", "리슬링": "2~4만원"}
    for k, v in prices.items():
        if k in wine:
            return f"{wine} 가격대: {v}"
    return f"{wine}: 가격 정보 없음"


agent = create_react_agent(
    llm, [wine_knowledge, price_lookup],
    prompt="당신은 와인 추천 에이전트입니다. 지식 검색과 가격 조회 도구를 활용해 답하세요.",
)

result = agent.invoke({"messages": [("human", "스테이크에 어울리는 와인과 그 가격대를 알려줘")]})
print(msg_text(result["messages"][-1].content)[:250])

print("\n✅ ch03 에이전트 + RAG 실습 완료")
