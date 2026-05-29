"""
랭체인과 RAG로 배우는 실전 LLM (위키북스) — GraphRAG 지식그래프 검색
강의 요소: GraphRAG 지식그래프 기반 검색
(PEFT 경량 파인튜닝은 GPU 필요 → 개념은 강의 슬라이드로, 코드는 GraphRAG 시연)

【교재→2026 업데이트】
  - Neo4j 등 그래프DB → 파이썬 dict 기반 경량 지식그래프 (외부 의존 없이 개념 실습)
  - 그래프에서 엔티티 이웃을 추출 → 컨텍스트로 LLM 종합
"""
import os
from dotenv import load_dotenv
load_dotenv()

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

llm = ChatGoogleGenerativeAI(model=os.getenv("GEMINI_MODEL", "gemini-2.5-flash"), temperature=0)

# 경량 지식그래프: (주어, 관계, 목적어)
TRIPLES = [
    ("랭체인", "개발자", "Harrison Chase"),
    ("랭체인", "기반언어", "파이썬"),
    ("랭그래프", "상위프레임워크", "랭체인"),
    ("랭그래프", "용도", "멀티에이전트 워크플로"),
    ("RAG", "구성요소", "리트리버"),
]


def graph_neighbors(entity: str):
    """엔티티와 직접 연결된 트리플(이웃)을 그래프에서 추출"""
    return [t for t in TRIPLES if entity in (t[0], t[2])]


prompt = ChatPromptTemplate.from_template(
    "다음 지식그래프 관계만 근거로 질문에 답하세요.\n[관계]\n{facts}\n[질문] {q}"
) | llm | StrOutputParser()


def graph_rag(question: str, entity: str):
    facts = "\n".join(f"{s} -[{r}]-> {o}" for s, r, o in graph_neighbors(entity))
    print(f"[추출된 '{entity}' 이웃 관계]\n{facts}")
    return prompt.invoke({"facts": facts, "q": question})


print(graph_rag("랭그래프는 무엇의 상위 프레임워크이고 어디에 쓰여?", "랭그래프").strip()[:160])

print("\n✅ ch03 GraphRAG 실습 완료")
