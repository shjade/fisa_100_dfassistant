from openai import OpenAI
from dotenv import load_dotenv
import os
import pandas as pd
import streamlit as st

load_dotenv() # .env를 읽어옵니다
api_key=os.getenv('OPEN_AI_KEY')

'''def load_data():
    df = pd.read_excel('Adidas US Sales Datasets.xlsx', skiprows=4) # skiprows=삭제할행개수
    df = df.drop("Unnamed: 0", axis=1)
    return df'''

@st.cache_data # 함수의 결과값이 이전과 같으면 캐싱된 데이터를 재사용
def load_data():
    df = pd.read_csv(r"C:\ITStudy\data\final_df.csv")
    return df

df = load_data()
# R F T C 를 지켜서 작성하면 결과가 좀더 예측가능하고 구체적으로 나온다 
# Role (역할)
# Format (형식)
# Task (해야할 일)
# Constraint(제약조건) or Context(문맥)
query = st.text_input('질문을 입력하세요: ')

def table_definition_prompt(df):
    prompt = '''Given the following pandas dataframe definition,
            write queries based on the request
            \n### pandas dataframe, with its properties:

            #
            # df의 컬럼명({})
            #
            '''.format(",".join(str(x) for x in df.columns))

    return prompt


client = OpenAI(api_key=api_key)

response = client.responses.create(
    model="gpt-5-nano",
    input=[
            {"role": "system", "content": "You are an assistant that generates Pandas boolean indexing code based on the given df definition\
            and a natural language request. The answer should start with df and contains only code by one line, not any explanation or ``` for copy."},
            {"role": "user", "content": f"A query to answer: {table_definition_prompt(df) + query}"}
        ]
)

st.code(response.output_text) # 믿을 수 없으니까 중간과정도 확인
st.dataframe(eval(response.output_text))
# 