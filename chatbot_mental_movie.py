import streamlit as st
from streamlit_chat import message # 설치(파이썬 3.8 이상에서만 작동)
import pandas as pd
from sentence_transformers import SentenceTransformer #sentence_transformer 설치 필요
from sklearn.metrics.pairwise import cosine_similarity
import json
from movie import recommend_movie

@st.cache_resource
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

st.header('심리상담 + 영화상담')
st.markdown("[Chatbot], 신희찬)")

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
    if user_input.startswith('/영화추천'):
        tmp_input = user_input.split(maxsplit=1) # split() 함수를 사용하여 입력 문자열을 공백을 기준으로 분할
        if len(tmp_input) >= 2 and tmp_input[0] == '/영화추천' and tmp_input[1] is not None:
            res = recommend_movie(tmp_input[1])
            if res:
                # 영화 추천 결과를 챗봇의 답변으로 그대로 추가
                st.session_state.generated.append(res)
            else:
                st.session_state.generated.append("영화를 찾을 수 없습니다.")
    else:
        # 영화 추천이 아닌 경우에도 모델을 사용하여 답변 생성
        embedding = model.encode(user_input)
        df['distance'] = df['embedding'].map(lambda x: cosine_similarity([embedding], [x]).squeeze())
        answer = df.loc[df['distance'].idxmax()]
        st.session_state.generated.append(answer['챗봇'])

    st.session_state.past.append(user_input)

for i in range(len(st.session_state['past'])):
    message(st.session_state['past'][i], is_user=True, key=str(i) + '_user')
    if len(st.session_state['generated']) > i:
        if isinstance(st.session_state['generated'][i], list):  # 만약 영화 추천 결과인 경우
            bot_response = "<br>".join([f"<b>{title}</b> ({score}) - {', '.join(genres)}" for title, score, genres in st.session_state['generated'][i]])
            st.markdown(f"<div style='background-color: #333333; color: white; padding: 10px; border-radius: 5px;'><b>Bot:</b><br>{bot_response}</div>", unsafe_allow_html=True)
        else:  # 일반적인 챗봇 응답인 경우
            st.markdown(f"<div style='background-color: #333333; color: white; padding: 10px; border-radius: 5px;'><b>Bot:</b> {st.session_state['generated'][i]}</div>", unsafe_allow_html=True)
