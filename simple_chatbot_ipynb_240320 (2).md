# SentenceTransformer

```python
!pip install sentence_transformers

import pandas as pd

# 문장을 고정된 길이의 벡터로 변환 필요 => SentenceTransformer 문장 임베딩 생성
from sentence_transformers import SentenceTransformer

# cosine_similarity 두 개의 행렬 또는 배열 간의 코사인 유사도를 계산 = 두 벡터간의 유사성 측정 (-1 ~ 1)
from sklearn.metrics.pairwise import cosine_similarity
```



# SentenceBERT 모델 로드

** Bert(Bidirectional Encoder Representations from Transformers), 레이블이 지정된 작업에 대해 추가적인 미세 조정(fine-tuning)이 가능
 **





```python
# 문장 단위로 텍스트를 인코딩하여 문장 간의 의미 유사성을 측정할 수 있는 방법을 제공하는 모델
model = SentenceTransformer('jhgan/ko-sroberta-multitask')

sentences = ["안녕하세요?", "한국어 문장 임베딩을 위한 버트 모델입니다."]
embeddings = model.encode(sentences)

print(embeddings)
```

    /usr/local/lib/python3.10/dist-packages/huggingface_hub/utils/_token.py:88: UserWarning: 
    The secret `HF_TOKEN` does not exist in your Colab secrets.
    To authenticate with the Hugging Face Hub, create a token in your settings tab (https://huggingface.co/settings/tokens), set it as secret in your Google Colab and restart your session.
    You will be able to reuse this secret in all of your notebooks.
    Please note that authentication is recommended but still optional to access public models or datasets.
      warnings.warn(



    modules.json:   0%|          | 0.00/229 [00:00<?, ?B/s]

    config_sentence_transformers.json:   0%|          | 0.00/123 [00:00<?, ?B/s]

    README.md:   0%|          | 0.00/4.86k [00:00<?, ?B/s]

    sentence_bert_config.json:   0%|          | 0.00/53.0 [00:00<?, ?B/s]

    config.json:   0%|          | 0.00/744 [00:00<?, ?B/s]

    pytorch_model.bin:   0%|          | 0.00/443M [00:00<?, ?B/s]

    tokenizer_config.json:   0%|          | 0.00/585 [00:00<?, ?B/s]

    vocab.txt:   0%|          | 0.00/248k [00:00<?, ?B/s]

    tokenizer.json:   0%|          | 0.00/495k [00:00<?, ?B/s]

    special_tokens_map.json:   0%|          | 0.00/156 [00:00<?, ?B/s]

    1_Pooling/config.json:   0%|          | 0.00/190 [00:00<?, ?B/s]


    [[-0.37510458 -0.77338415  0.5927711  ...  0.5792351   0.32683465
      -0.65089613]
     [-0.0936174  -0.1819152  -0.19230822 ... -0.03165796  0.30412537
      -0.26793632]]



# 데이터셋 로드



**본 데이터셋은 웰니스 대화 스크립트 데이터셋으로 심리에 대한 상담내용을 담은 데이터셋이다.**

https://aihub.or.kr/opendata/keti-data/recognition-laguage/KETI-02-006




```python
df = pd.read_csv('/content/drive/MyDrive/Dataset/Chatbot/wellness_dataset_original.csv')
df.head()
```

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>구분</th>
      <th>유저</th>
      <th>챗봇</th>
      <th>Unnamed: 3</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>감정/감정조절이상</td>
      <td>제 감정이 이상해진 것 같아요. 남편만 보면 화가 치밀어 오르고 감정 조절이 안되요.</td>
      <td>감정이 조절이 안 될 때만큼 힘들 때는 없는 거 같아요.</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>감정/감정조절이상</td>
      <td>더 이상 내 감정을 내가 컨트롤 못 하겠어.</td>
      <td>저도 그 기분 이해해요. 많이 힘드시죠?</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>감정/감정조절이상</td>
      <td>하루종일 오르락내리락 롤러코스터 타는 기분이에요.</td>
      <td>그럴 때는 밥은 잘 먹었는지, 잠은 잘 잤는지 체크해보는 것도 좋아요.</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>감정/감정조절이상</td>
      <td>꼭 롤러코스터 타는 것 같아요.</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4</th>
      <td>감정/감정조절이상</td>
      <td>롤러코스터 타는 것처럼 기분이 왔다 갔다 해요.</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>




# 전처리




```python
# Nan 제거
df = df.drop(columns=['Unnamed: 3'])

df.head()
```

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>구분</th>
      <th>유저</th>
      <th>챗봇</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>감정/감정조절이상</td>
      <td>제 감정이 이상해진 것 같아요. 남편만 보면 화가 치밀어 오르고 감정 조절이 안되요.</td>
      <td>감정이 조절이 안 될 때만큼 힘들 때는 없는 거 같아요.</td>
    </tr>
    <tr>
      <th>1</th>
      <td>감정/감정조절이상</td>
      <td>더 이상 내 감정을 내가 컨트롤 못 하겠어.</td>
      <td>저도 그 기분 이해해요. 많이 힘드시죠?</td>
    </tr>
    <tr>
      <th>2</th>
      <td>감정/감정조절이상</td>
      <td>하루종일 오르락내리락 롤러코스터 타는 기분이에요.</td>
      <td>그럴 때는 밥은 잘 먹었는지, 잠은 잘 잤는지 체크해보는 것도 좋아요.</td>
    </tr>
    <tr>
      <th>3</th>
      <td>감정/감정조절이상</td>
      <td>꼭 롤러코스터 타는 것 같아요.</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4</th>
      <td>감정/감정조절이상</td>
      <td>롤러코스터 타는 것처럼 기분이 왔다 갔다 해요.</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>


```python
# Nan이 아닌 행 추출
df = df[~df['챗봇'].isna()]

df.head()
```

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>구분</th>
      <th>유저</th>
      <th>챗봇</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>감정/감정조절이상</td>
      <td>제 감정이 이상해진 것 같아요. 남편만 보면 화가 치밀어 오르고 감정 조절이 안되요.</td>
      <td>감정이 조절이 안 될 때만큼 힘들 때는 없는 거 같아요.</td>
    </tr>
    <tr>
      <th>1</th>
      <td>감정/감정조절이상</td>
      <td>더 이상 내 감정을 내가 컨트롤 못 하겠어.</td>
      <td>저도 그 기분 이해해요. 많이 힘드시죠?</td>
    </tr>
    <tr>
      <th>2</th>
      <td>감정/감정조절이상</td>
      <td>하루종일 오르락내리락 롤러코스터 타는 기분이에요.</td>
      <td>그럴 때는 밥은 잘 먹었는지, 잠은 잘 잤는지 체크해보는 것도 좋아요.</td>
    </tr>
    <tr>
      <th>15</th>
      <td>감정/감정조절이상/화</td>
      <td>평소 다른 일을 할 때도 비슷해요. 생각한대로 안되면 화가 나고…그런 상황이 지속되...</td>
      <td>화가 폭발할 것 같을 때는 그 자리를 피하는 것도 좋은 방법이라고 생각해요.</td>
    </tr>
    <tr>
      <th>16</th>
      <td>감정/감정조절이상/화</td>
      <td>예전보다 화내는 게 과격해진 거 같아.</td>
      <td>정말 힘드시겠어요. 화는 남에게도 스스로에게도 상처를 주잖아요.</td>
    </tr>
  </tbody>
</table>


```python
df.loc[0, '유저']
```


    '제 감정이 이상해진 것 같아요. 남편만 보면 화가 치밀어 오르고 감정 조절이 안되요.'




```python
model.encode(df.loc[0, '유저'])
```


    array([-4.80606765e-01, -2.94868946e-01,  4.37900245e-01, -6.40137374e-01,
            3.28670219e-02, -3.42647523e-01, -5.47481887e-02,  1.73054636e-02,
           -4.08221185e-01, -5.06034195e-01, -1.68733329e-01, -3.98677349e-01,
           -1.24776624e-01, -9.71540883e-02, -1.65286273e-01,  5.72613114e-03,
    		...
            1.28170326e-01, -1.43580198e-01,  2.30380893e-01, -5.46675883e-02,
            3.71279776e-01,  1.98934287e-01,  4.64870483e-01,  3.64101559e-01,
           -4.83271256e-02,  2.67422974e-01, -7.10749209e-01,  5.07541671e-02],
          dtype=float32)



# 유저 대화내용 인코딩



**유저의 모든 발화를 임베딩에 저장한다.**


```python
df['embedding'] = pd.Series([[]] * len(df)) # dummy

df['embedding'] = df['유저'].map(lambda x: list(model.encode(x)))

df.head()
```

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>구분</th>
      <th>유저</th>
      <th>챗봇</th>
      <th>embedding</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>감정/감정조절이상</td>
      <td>제 감정이 이상해진 것 같아요. 남편만 보면 화가 치밀어 오르고 감정 조절이 안되요.</td>
      <td>감정이 조절이 안 될 때만큼 힘들 때는 없는 거 같아요.</td>
      <td>[-0.48060676, -0.29486895, 0.43790025, -0.6401...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>감정/감정조절이상</td>
      <td>더 이상 내 감정을 내가 컨트롤 못 하겠어.</td>
      <td>저도 그 기분 이해해요. 많이 힘드시죠?</td>
      <td>[-1.1561574, -0.14506245, 0.29490346, -0.67394...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>감정/감정조절이상</td>
      <td>하루종일 오르락내리락 롤러코스터 타는 기분이에요.</td>
      <td>그럴 때는 밥은 잘 먹었는지, 잠은 잘 잤는지 체크해보는 것도 좋아요.</td>
      <td>[-0.66520053, -0.081268094, 1.0945567, 0.10579...</td>
    </tr>
    <tr>
      <th>15</th>
      <td>감정/감정조절이상/화</td>
      <td>평소 다른 일을 할 때도 비슷해요. 생각한대로 안되면 화가 나고…그런 상황이 지속되...</td>
      <td>화가 폭발할 것 같을 때는 그 자리를 피하는 것도 좋은 방법이라고 생각해요.</td>
      <td>[-0.767906, 0.465207, 0.5285069, -0.50760436, ...</td>
    </tr>
    <tr>
      <th>16</th>
      <td>감정/감정조절이상/화</td>
      <td>예전보다 화내는 게 과격해진 거 같아.</td>
      <td>정말 힘드시겠어요. 화는 남에게도 스스로에게도 상처를 주잖아요.</td>
      <td>[-0.20277722, -0.37413904, 0.040531933, -0.862...</td>
    </tr>
  </tbody>
</table>


```python
# 데이터프레임을 CSV 파일로 저장

df.to_csv('wellness_dataset.csv', index=False)
```



# SIMPLE 챗봇 구현




```python
# 입력 예시안

text = '요즘 머리가 아프고 너무 힘들어'
embedding = model.encode(text)
```


```python
# 유저가 쓴 문장과 코사인 유사도가 높은 값을 찾아 임베딩한 가장 가까운 결과와의 거리를 구한 후 distance 변수에 넣어준다.
# 즉, distance가 1에 가장 가까운 문장이 출력되는 구조이다.

df['distance'] = df['embedding'].map(lambda x: cosine_similarity([embedding], [x]).squeeze()) 

df.head()
```

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>구분</th>
      <th>유저</th>
      <th>챗봇</th>
      <th>embedding</th>
      <th>distance</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>감정/감정조절이상</td>
      <td>제 감정이 이상해진 것 같아요. 남편만 보면 화가 치밀어 오르고 감정 조절이 안되요.</td>
      <td>감정이 조절이 안 될 때만큼 힘들 때는 없는 거 같아요.</td>
      <td>[-0.48060676, -0.29486895, 0.43790025, -0.6401...</td>
      <td>0.448967</td>
    </tr>
    <tr>
      <th>1</th>
      <td>감정/감정조절이상</td>
      <td>더 이상 내 감정을 내가 컨트롤 못 하겠어.</td>
      <td>저도 그 기분 이해해요. 많이 힘드시죠?</td>
      <td>[-1.1561574, -0.14506245, 0.29490346, -0.67394...</td>
      <td>0.490199</td>
    </tr>
    <tr>
      <th>2</th>
      <td>감정/감정조절이상</td>
      <td>하루종일 오르락내리락 롤러코스터 타는 기분이에요.</td>
      <td>그럴 때는 밥은 잘 먹었는지, 잠은 잘 잤는지 체크해보는 것도 좋아요.</td>
      <td>[-0.66520053, -0.081268094, 1.0945567, 0.10579...</td>
      <td>0.352131</td>
    </tr>
    <tr>
      <th>15</th>
      <td>감정/감정조절이상/화</td>
      <td>평소 다른 일을 할 때도 비슷해요. 생각한대로 안되면 화가 나고…그런 상황이 지속되...</td>
      <td>화가 폭발할 것 같을 때는 그 자리를 피하는 것도 좋은 방법이라고 생각해요.</td>
      <td>[-0.767906, 0.465207, 0.5285069, -0.50760436, ...</td>
      <td>0.422284</td>
    </tr>
    <tr>
      <th>16</th>
      <td>감정/감정조절이상/화</td>
      <td>예전보다 화내는 게 과격해진 거 같아.</td>
      <td>정말 힘드시겠어요. 화는 남에게도 스스로에게도 상처를 주잖아요.</td>
      <td>[-0.20277722, -0.37413904, 0.040531933, -0.862...</td>
      <td>0.315118</td>
    </tr>
  </tbody>
</table>


```python
# 최대값 추출

answer = df.loc[df['distance'].idxmax()]

print('구분 :', answer['구분'])
print('유사한 질문 :', answer['유저'])
print('챗봇 답변 :', answer['챗봇'])
print('유사도 :', answer['distance'])
```

    구분 : 증상/편두통
    유사한 질문 : 요즘은 머리가 한쪽만 지그시 누르는 것처럼 무겁고 아파요.
    챗봇 답변 : 으으, 머리가 아프면 정말 힘들죠. 그 마음 정말 이해해요.
    유사도 : 0.8296288251876831

