"""
LangGraph로 만드는 AI 에이전트 서비스 — RAG 통합 에이전트
강의 요소: RAG 통합

【교재→2026 업데이트】
  - create_react_agent + 검색 도구로 RAG를 에이전트에 통합
  - GoogleGenerativeAIEmbeddings + Chroma 인메모리
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


FAQ = [
    "환불은 구매 후 14일 이내 가능하며 영수증이 필요합니다.",
    "배송은 주문 후 평균 2~3일 소요됩니다.",
    "멤버십은 연 3만원이며 무료 배송 혜택이 있습니다.",
]
retriever = Chroma.from_documents([Document(page_content=t) for t in FAQ],
                                  embedding=embeddings).as_retriever(search_kwargs={"k": 1})


@tool
def search_faq(query: str) -> str:
    """고객센터 FAQ에서 관련 답변을 검색합니다."""
    return retriever.invoke(query)[0].page_content


agent = create_react_agent(llm, [search_faq],
                           prompt="고객센터 상담원입니다. FAQ 검색 도구로 정확히 답하세요.")
result = agent.invoke({"messages": [("human", "환불 기간이 어떻게 되나요?")]})
print(msg_text(result["messages"][-1].content)[:150])

print("\n✅ ch02 RAG 통합 에이전트 실습 완료")
