"""
랭체인과 RAG로 배우는 실전 LLM (위키북스) — 기본 RAG 파이프라인
강의 요소: RAG 파이프라인(임베딩·벡터스토어·검색)

【교재→2026 업데이트】
  - 코랩/OpenAI → 로컬 CPU + Gemini (gemini-2.5-flash, models/gemini-embedding-2)
  - RetrievalQA → LCEL RAG 체인
"""
import os
from dotenv import load_dotenv
load_dotenv()

from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

llm = ChatGoogleGenerativeAI(model=os.getenv("GEMINI_MODEL", "gemini-2.5-flash"), temperature=0)
embeddings = GoogleGenerativeAIEmbeddings(model=os.getenv("GEMINI_EMBEDDING_MODEL", "models/gemini-embedding-2"))

TEXT = ("ReAct는 추론(Reasoning)과 행동(Acting)을 번갈아 수행하는 에이전트 기법이다. "
        "GraphRAG는 지식그래프로 엔티티 관계를 활용해 검색한다. "
        "PEFT는 적은 파라미터만 학습하는 경량 파인튜닝 기법이다. "
        "RAG는 검색으로 외부 지식을 보강한다.")
splits = RecursiveCharacterTextSplitter(chunk_size=45, chunk_overlap=10).split_documents([Document(page_content=TEXT)])
retriever = Chroma.from_documents(splits, embedding=embeddings).as_retriever(search_kwargs={"k": 2})

prompt = ChatPromptTemplate.from_template("컨텍스트로만 답하세요.\n{context}\n질문: {question}")
rag = ({"context": retriever | (lambda ds: "\n".join(d.page_content for d in ds)),
        "question": RunnablePassthrough()} | prompt | llm | StrOutputParser())

for q in ["ReAct가 뭐야?", "PEFT의 특징은?"]:
    print(f"Q: {q}\nA: {rag.invoke(q).strip()[:120]}\n")

print("✅ ch01 기본 RAG 파이프라인 실습 완료")
