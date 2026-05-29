"""
테디노트 RAG 비법노트 — 기본편 — 4. 8단계 RAG 체인 완성
(원본 저장소: github.com/teddylee777/langchain-kr — RAG 기본 파이프라인)

8단계: ①문서 로드 ②분할 ③임베딩 ④벡터 저장 ⑤검색 ⑥프롬프트 ⑦LLM ⑧체인(LCEL)

【교재→2026 업데이트】
  - ChatOpenAI + OpenAIEmbeddings → ChatGoogleGenerativeAI + GoogleGenerativeAIEmbeddings
  - RetrievalQA(체인 클래스) → LCEL 명시적 RAG 체인 (retriever | format | prompt | llm | parser)
  - models/gemini-embedding-2 임베딩, gemini-2.5-flash 생성
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

EMBEDDING_MODEL = os.getenv("GEMINI_EMBEDDING_MODEL", "models/gemini-embedding-2")
llm = ChatGoogleGenerativeAI(model=os.getenv("GEMINI_MODEL", "gemini-2.5-flash"), temperature=0)
embeddings = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL)

# ① 문서 로드 (인라인 더미 — 실제 환경에선 PyPDFLoader 등)
raw = (
    "테디노트 RAG 비법노트 기본편은 8단계 RAG 파이프라인을 다룹니다. "
    "RAG는 검색 증강 생성을 뜻하며, LLM이 모르는 최신/사내 지식을 외부 문서에서 검색해 보강합니다. "
    "LCEL은 LangChain Expression Language로, 파이프(|) 연산으로 체인을 구성합니다. "
    "PART06에서는 Streamlit으로 RAG 챗봇 웹앱을 만들어 배포합니다. "
    "임베딩 모델은 텍스트를 벡터로 바꾸고, 벡터 스토어가 유사도 검색을 담당합니다."
)
docs = [Document(page_content=raw, metadata={"source": "teddynote-basic"})]

# ② 분할
splitter = RecursiveCharacterTextSplitter(chunk_size=70, chunk_overlap=15)
splits = splitter.split_documents(docs)

# ③④ 임베딩 + 벡터 저장
vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)

# ⑤ 검색
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# ⑥ 프롬프트
prompt = ChatPromptTemplate.from_template(
    "아래 컨텍스트만 근거로 한국어로 간결히 답하세요. 모르면 모른다고 하세요.\n\n"
    "[컨텍스트]\n{context}\n\n[질문]\n{question}\n\n[답변]"
)

def format_docs(retrieved):
    return "\n".join(d.page_content for d in retrieved)

# ⑦⑧ LLM + 체인 (LCEL)
rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

print("=== 8단계 RAG 체인 실행 ===")
for q in ["RAG가 뭐야?", "LCEL은 무엇이고 어떻게 체인을 만들지?", "PART06에서는 무엇을 만들어?"]:
    print(f"\nQ: {q}")
    print(f"A: {rag_chain.invoke(q).strip()[:200]}")

print("\n✅ ch04 8단계 RAG 체인 실습 완료")
