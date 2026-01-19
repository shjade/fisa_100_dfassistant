import os
import ast
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI

# -----------------------------
# 기본 설정
# -----------------------------
st.set_page_config(page_title="DF Query (OpenAI)", layout="wide")
st.title("Adidas 판매 데이터 자동 답변 서비스")

# -----------------------------
# .env 로드 + API 키
# -----------------------------
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    st.error("OPENAI_API_KEY가 없습니다. .env에 OPENAI_API_KEY=... 형태로 넣어주세요.")
    st.stop()

client = OpenAI(api_key=api_key)

# -----------------------------
# ✅ df 로드 
# -----------------------------
DATA_PATH = r"C:\ITStudy\data\final_df.csv"

try:
    df = pd.read_csv(DATA_PATH)
except Exception as e:
    st.error("데이터를 불러오지 못했습니다.")
    st.write(f"경로: {DATA_PATH}")
    st.exception(e)
    st.stop()

#st.success(f"데이터 로드 완료: {df.shape[0]:,} rows × {df.shape[1]} cols")
#st.dataframe(df.head(20), use_container_width=True)

# -----------------------------
# 프롬프트 생성 (네 코드 유지)
# -----------------------------
def table_definition_prompt(df: pd.DataFrame) -> str:
    prompt = """Given the following pandas dataframe definition,
write queries based on the request

### pandas dataframe, with its properties:
# df의 컬럼명({})
#""".format(",".join(str(x) for x in df.columns))
    return prompt

SYSTEM_MSG = (
    "You are an assistant that generates Pandas boolean indexing code based on the given df definition "
    "and a natural language request. The answer should start with df and contains only code by one line, "
    "not any explanation or ``` for copy."
)

# eval 최소 방어 (df만 허용)
def safe_eval_df_only(code: str, df: pd.DataFrame):
    ast.parse(code, mode="eval")
    return eval(code, {"__builtins__": {}}, {"df": df})

# -----------------------------
# Streamlit input / output
# -----------------------------
question = st.text_input(
    "질문을 입력하세요",
    placeholder="예 : Newyork 구매 건수 알려줘"
)

mode = st.radio("동작", ["코드 생성", "코드 생성 + 실행(eval)"], index=0)

if st.button("답변 확인", type="primary", disabled=not question.strip()):
    user_prompt = table_definition_prompt(df) + question

    with st.spinner("답변 준비중..."):
        response = client.responses.create(
            model="gpt-5-nano",
            input=[
                {"role": "system", "content": SYSTEM_MSG},
                {"role": "user", "content": f"A query to answer: {user_prompt}"},
            ],
        )

    code = (response.output_text or "").strip()

    st.subheader("답변 도출을위한 코드")
    st.code(code, language="python")

    if mode == "코드 생성 + 실행(eval)":
        try:
            result = safe_eval_df_only(code, df)
            st.subheader("✅ 실행 결과")
            if isinstance(result, (pd.DataFrame, pd.Series)):
                st.dataframe(result.head(200), use_container_width=True)
            else:
                st.write(result)
        except Exception as e:
            st.error("실행 실패")
            st.exception(e)
