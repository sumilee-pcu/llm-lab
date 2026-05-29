"""
LangChain을 활용한 Google Cloud의 생성 AI — 자연어 → SQL
강의 요소: 에이전트·자연어→SQL

【교재→2026 업데이트】
  - BigQuery / Cloud SQL → 로컬 SQLite 인메모리 (외부 클라우드 없이 동일 개념 실습)
  - LLM이 스키마를 보고 SQL 생성 → 실제 실행 → 결과 자연어화
"""
import os, sqlite3, re
from dotenv import load_dotenv
load_dotenv()

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

llm = ChatGoogleGenerativeAI(model=os.getenv("GEMINI_MODEL", "gemini-2.5-flash"), temperature=0)

# 1. 인메모리 DB 준비
conn = sqlite3.connect(":memory:")
conn.executescript("""
CREATE TABLE sales(id INTEGER, product TEXT, region TEXT, amount INTEGER);
INSERT INTO sales VALUES (1,'노트북','서울',1200),(2,'마우스','부산',30),
(3,'노트북','부산',1100),(4,'키보드','서울',80),(5,'노트북','서울',1300);
""")
SCHEMA = "테이블 sales(id INTEGER, product TEXT, region TEXT, amount INTEGER)"

sql_chain = ChatPromptTemplate.from_template(
    "SQLite 스키마: {schema}\n질문: {q}\n"
    "질문에 답하는 SQL 한 줄만 출력하세요. 코드블록·설명 없이 SQL만."
) | llm | StrOutputParser()


def run_nl_query(q: str):
    raw = sql_chain.invoke({"schema": SCHEMA, "q": q}).strip()
    sql = re.sub(r"```sql|```", "", raw).strip().rstrip(";")
    rows = conn.execute(sql).fetchall()
    return sql, rows


for q in ["서울 지역 노트북 총 매출은?", "지역별 매출 합계를 큰 순서로"]:
    sql, rows = run_nl_query(q)
    print(f"Q: {q}\n  SQL: {sql}\n  결과: {rows}\n")

print("✅ ch03 자연어→SQL 실습 완료")
