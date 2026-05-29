"""
LLM 개발 애플리케이션 — RAG 프로젝트 실습
강의 요소: RAG 프로젝트 (임베딩→검색→생성)

【교재→2026 업데이트】
  - ChatOpenAI + OpenAIEmbeddings → ChatGoogleGenerativeAI + GoogleGenerativeAIEmbeddings
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

KNOWLEDGE = (
    "트랜스포머는 2017년 'Attention Is All You Need' 논문에서 제안되었습니다. "
    "BERT는 인코더 기반 양방향 모델로 문장 이해에 강합니다. "
    "GPT는 디코더 기반 자기회귀 모델로 텍스트 생성에 강합니다. "
    "T5는 모든 NLP 과제를 텍스트-투-텍스트로 통일했습니다. "
    "RAG는 검색으로 외부 지식을 보강해 환각을 줄입니다."
)
splits = RecursiveCharacterTextSplitter(chunk_size=50, chunk_overlap=10).split_documents(
    [Document(page_content=KNOWLEDGE)])
retriever = Chroma.from_documents(splits, embedding=embeddings).as_retriever(search_kwargs={"k": 2})

prompt = ChatPromptTemplate.from_template(
    "컨텍스트만 근거로 한국어로 답하세요.\n[컨텍스트]\n{context}\n\n[질문] {question}")
rag = ({"context": retriever | (lambda ds: "\n".join(d.page_content for d in ds)),
        "question": RunnablePassthrough()} | prompt | llm | StrOutputParser())

for q in ["BERT와 GPT의 차이는?", "RAG는 왜 환각을 줄이지?"]:
    print(f"Q: {q}\nA: {rag.invoke(q).strip()[:150]}\n")

print("✅ ch02 RAG 프로젝트 실습 완료")
