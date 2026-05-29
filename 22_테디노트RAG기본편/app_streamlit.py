"""
테디노트 RAG 비법노트 — 기본편 — 5. Streamlit RAG 챗봇 웹앱 (PART06 대응)
(원본 저장소: github.com/teddylee777/langchain-kr — PART06 Streamlit)

실행:  streamlit run app_streamlit.py
※ 인터랙티브 앱이라 일반 `python` 실행이 아닌 streamlit 런처로 구동합니다.

【교재→2026 업데이트】
  - ChatOpenAI + OpenAIEmbeddings → ChatGoogleGenerativeAI + GoogleGenerativeAIEmbeddings
  - st.cache_resource로 벡터 스토어/체인 캐싱 (1.x 권장)
  - LCEL 기반 RAG 체인 (ch04와 동일 파이프라인)
"""
import os
from dotenv import load_dotenv
load_dotenv()

import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

EMBEDDING_MODEL = os.getenv("GEMINI_EMBEDDING_MODEL", "models/gemini-embedding-2")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

KNOWLEDGE = (
    "RAG는 검색 증강 생성입니다. 외부 문서를 검색해 LLM 답변 근거로 씁니다. "
    "8단계 파이프라인: 로드·분할·임베딩·저장·검색·프롬프트·LLM·체인. "
    "LCEL은 prompt | llm | parser 로 체인을 구성합니다. "
    "이 앱은 Streamlit으로 만든 RAG 챗봇 데모입니다."
)


@st.cache_resource
def build_chain():
    splits = RecursiveCharacterTextSplitter(chunk_size=70, chunk_overlap=15).split_documents(
        [Document(page_content=KNOWLEDGE, metadata={"source": "demo"})]
    )
    embeddings = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL)
    retriever = Chroma.from_documents(splits, embedding=embeddings).as_retriever(
        search_kwargs={"k": 3}
    )
    llm = ChatGoogleGenerativeAI(model=GEMINI_MODEL, temperature=0)
    prompt = ChatPromptTemplate.from_template(
        "아래 컨텍스트만 근거로 한국어로 답하세요.\n[컨텍스트]\n{context}\n\n[질문]\n{question}"
    )
    return (
        {"context": retriever | (lambda ds: "\n".join(d.page_content for d in ds)),
         "question": RunnablePassthrough()}
        | prompt | llm | StrOutputParser()
    )


def main():
    st.set_page_config(page_title="테디노트 RAG 챗봇", page_icon="📓")
    st.title("📓 테디노트 RAG 기본편 — 챗봇 데모")
    chain = build_chain()

    if "messages" not in st.session_state:
        st.session_state.messages = []
    for m in st.session_state.messages:
        st.chat_message(m["role"]).write(m["content"])

    if q := st.chat_input("질문을 입력하세요 (예: RAG가 뭐야?)"):
        st.chat_message("user").write(q)
        st.session_state.messages.append({"role": "user", "content": q})
        answer = chain.invoke(q)
        st.chat_message("assistant").write(answer)
        st.session_state.messages.append({"role": "assistant", "content": answer})


if __name__ == "__main__":
    main()
