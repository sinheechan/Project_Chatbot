import streamlit as st
from streamlit_chat import message # 설치(파이썬 3.8 이상에서만 작동)
import pandas as pd
from sentence_transformers import SentenceTransformer #sentence_transformer 설치 필요
from sklearn.metrics.pairwise import cosine_similarity
import json

@ st.cache_resource
def cached_model():
    model = SentenceTransformer('jhgan/ko-sroberta-multitask')
    return model

@st.cache_resource
def get_dataset():
    df = pd.read_csv('wellness_dataset.csv')
    df['embedding'] = df['embedding'].apply(json.loads)
    return df

model = cached_model()
df = get_dataset()

st.header('심리상담 챗봇')
st.markdown("[코리아 IT 아카데미], 이석창 강사)")

# 대화한 내용 저장
if 'generated' not in st.session_state:
    st.session_state['generated'] = []

# 지난 대화 저장
if 'past' not in st.session_state:
    st.session_state['past'] = []

# 사용자 입력 폼
with st.form('form', clear_on_submit=True):
    user_input = st.text_input('당신: ', '')
    submitted = st.form_submit_button('전송')

if submitted and user_input:
    embedding = model.encode(user_input)

    df['distance'] = df['embedding'].map(lambda x: cosine_similarity([embedding], [x]).squeeze())
    answer = df.loc[df['distance'].idxmax()]

    st.session_state.past.append(user_input)
    st.session_state.generated.append(answer['챗봇'])

for i in range(len(st.session_state['past'])):
    message(st.session_state['past'][i], is_user=True, key=str(i) + '_user')
    if len(st.session_state['generated']) > i:
        message(st.session_state['generated'][i], key=str(i) + '_bot')
