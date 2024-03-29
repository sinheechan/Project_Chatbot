```python
!pip install sentence_transformers

import pandas as pd

# 문장을 고정된 길이의 벡터로 변환 필요 => SentenceTransformer 문장 임베딩 생성
from sentence_transformers import SentenceTransformer

# cosine_similarity 두 개의 행렬 또는 배열 간의 코사인 유사도를 계산 = 두 벡터간의 유사성 측정 (-1 ~ 1)
from sklearn.metrics.pairwise import cosine_similarity
```

    Collecting sentence_transformers
      Downloading sentence_transformers-2.5.1-py3-none-any.whl (156 kB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m156.5/156.5 kB[0m [31m3.5 MB/s[0m eta [36m0:00:00[0m
    [?25hRequirement already satisfied: transformers<5.0.0,>=4.32.0 in /usr/local/lib/python3.10/dist-packages (from sentence_transformers) (4.38.2)
    Requirement already satisfied: tqdm in /usr/local/lib/python3.10/dist-packages (from sentence_transformers) (4.66.2)
    Requirement already satisfied: torch>=1.11.0 in /usr/local/lib/python3.10/dist-packages (from sentence_transformers) (2.2.1+cu121)
    Requirement already satisfied: numpy in /usr/local/lib/python3.10/dist-packages (from sentence_transformers) (1.25.2)
    Requirement already satisfied: scikit-learn in /usr/local/lib/python3.10/dist-packages (from sentence_transformers) (1.2.2)
    Requirement already satisfied: scipy in /usr/local/lib/python3.10/dist-packages (from sentence_transformers) (1.11.4)
    Requirement already satisfied: huggingface-hub>=0.15.1 in /usr/local/lib/python3.10/dist-packages (from sentence_transformers) (0.20.3)
    Requirement already satisfied: Pillow in /usr/local/lib/python3.10/dist-packages (from sentence_transformers) (9.4.0)
    Requirement already satisfied: filelock in /usr/local/lib/python3.10/dist-packages (from huggingface-hub>=0.15.1->sentence_transformers) (3.13.1)
    Requirement already satisfied: fsspec>=2023.5.0 in /usr/local/lib/python3.10/dist-packages (from huggingface-hub>=0.15.1->sentence_transformers) (2023.6.0)
    Requirement already satisfied: requests in /usr/local/lib/python3.10/dist-packages (from huggingface-hub>=0.15.1->sentence_transformers) (2.31.0)
    Requirement already satisfied: pyyaml>=5.1 in /usr/local/lib/python3.10/dist-packages (from huggingface-hub>=0.15.1->sentence_transformers) (6.0.1)
    Requirement already satisfied: typing-extensions>=3.7.4.3 in /usr/local/lib/python3.10/dist-packages (from huggingface-hub>=0.15.1->sentence_transformers) (4.10.0)
    Requirement already satisfied: packaging>=20.9 in /usr/local/lib/python3.10/dist-packages (from huggingface-hub>=0.15.1->sentence_transformers) (24.0)
    Requirement already satisfied: sympy in /usr/local/lib/python3.10/dist-packages (from torch>=1.11.0->sentence_transformers) (1.12)
    Requirement already satisfied: networkx in /usr/local/lib/python3.10/dist-packages (from torch>=1.11.0->sentence_transformers) (3.2.1)
    Requirement already satisfied: jinja2 in /usr/local/lib/python3.10/dist-packages (from torch>=1.11.0->sentence_transformers) (3.1.3)
    Collecting nvidia-cuda-nvrtc-cu12==12.1.105 (from torch>=1.11.0->sentence_transformers)
      Downloading nvidia_cuda_nvrtc_cu12-12.1.105-py3-none-manylinux1_x86_64.whl (23.7 MB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m23.7/23.7 MB[0m [31m48.9 MB/s[0m eta [36m0:00:00[0m
    [?25hCollecting nvidia-cuda-runtime-cu12==12.1.105 (from torch>=1.11.0->sentence_transformers)
      Downloading nvidia_cuda_runtime_cu12-12.1.105-py3-none-manylinux1_x86_64.whl (823 kB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m823.6/823.6 kB[0m [31m67.2 MB/s[0m eta [36m0:00:00[0m
    [?25hCollecting nvidia-cuda-cupti-cu12==12.1.105 (from torch>=1.11.0->sentence_transformers)
      Downloading nvidia_cuda_cupti_cu12-12.1.105-py3-none-manylinux1_x86_64.whl (14.1 MB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m14.1/14.1 MB[0m [31m95.4 MB/s[0m eta [36m0:00:00[0m
    [?25hCollecting nvidia-cudnn-cu12==8.9.2.26 (from torch>=1.11.0->sentence_transformers)
      Downloading nvidia_cudnn_cu12-8.9.2.26-py3-none-manylinux1_x86_64.whl (731.7 MB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m731.7/731.7 MB[0m [31m2.1 MB/s[0m eta [36m0:00:00[0m
    [?25hCollecting nvidia-cublas-cu12==12.1.3.1 (from torch>=1.11.0->sentence_transformers)
      Downloading nvidia_cublas_cu12-12.1.3.1-py3-none-manylinux1_x86_64.whl (410.6 MB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m410.6/410.6 MB[0m [31m2.7 MB/s[0m eta [36m0:00:00[0m
    [?25hCollecting nvidia-cufft-cu12==11.0.2.54 (from torch>=1.11.0->sentence_transformers)
      Downloading nvidia_cufft_cu12-11.0.2.54-py3-none-manylinux1_x86_64.whl (121.6 MB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m121.6/121.6 MB[0m [31m8.7 MB/s[0m eta [36m0:00:00[0m
    [?25hCollecting nvidia-curand-cu12==10.3.2.106 (from torch>=1.11.0->sentence_transformers)
      Downloading nvidia_curand_cu12-10.3.2.106-py3-none-manylinux1_x86_64.whl (56.5 MB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m56.5/56.5 MB[0m [31m13.4 MB/s[0m eta [36m0:00:00[0m
    [?25hCollecting nvidia-cusolver-cu12==11.4.5.107 (from torch>=1.11.0->sentence_transformers)
      Downloading nvidia_cusolver_cu12-11.4.5.107-py3-none-manylinux1_x86_64.whl (124.2 MB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m124.2/124.2 MB[0m [31m8.2 MB/s[0m eta [36m0:00:00[0m
    [?25hCollecting nvidia-cusparse-cu12==12.1.0.106 (from torch>=1.11.0->sentence_transformers)
      Downloading nvidia_cusparse_cu12-12.1.0.106-py3-none-manylinux1_x86_64.whl (196.0 MB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m196.0/196.0 MB[0m [31m6.7 MB/s[0m eta [36m0:00:00[0m
    [?25hCollecting nvidia-nccl-cu12==2.19.3 (from torch>=1.11.0->sentence_transformers)
      Downloading nvidia_nccl_cu12-2.19.3-py3-none-manylinux1_x86_64.whl (166.0 MB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m166.0/166.0 MB[0m [31m7.0 MB/s[0m eta [36m0:00:00[0m
    [?25hCollecting nvidia-nvtx-cu12==12.1.105 (from torch>=1.11.0->sentence_transformers)
      Downloading nvidia_nvtx_cu12-12.1.105-py3-none-manylinux1_x86_64.whl (99 kB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m99.1/99.1 kB[0m [31m16.1 MB/s[0m eta [36m0:00:00[0m
    [?25hRequirement already satisfied: triton==2.2.0 in /usr/local/lib/python3.10/dist-packages (from torch>=1.11.0->sentence_transformers) (2.2.0)
    Collecting nvidia-nvjitlink-cu12 (from nvidia-cusolver-cu12==11.4.5.107->torch>=1.11.0->sentence_transformers)
      Downloading nvidia_nvjitlink_cu12-12.4.99-py3-none-manylinux2014_x86_64.whl (21.1 MB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m21.1/21.1 MB[0m [31m77.9 MB/s[0m eta [36m0:00:00[0m
    [?25hRequirement already satisfied: regex!=2019.12.17 in /usr/local/lib/python3.10/dist-packages (from transformers<5.0.0,>=4.32.0->sentence_transformers) (2023.12.25)
    Requirement already satisfied: tokenizers<0.19,>=0.14 in /usr/local/lib/python3.10/dist-packages (from transformers<5.0.0,>=4.32.0->sentence_transformers) (0.15.2)
    Requirement already satisfied: safetensors>=0.4.1 in /usr/local/lib/python3.10/dist-packages (from transformers<5.0.0,>=4.32.0->sentence_transformers) (0.4.2)
    Requirement already satisfied: joblib>=1.1.1 in /usr/local/lib/python3.10/dist-packages (from scikit-learn->sentence_transformers) (1.3.2)
    Requirement already satisfied: threadpoolctl>=2.0.0 in /usr/local/lib/python3.10/dist-packages (from scikit-learn->sentence_transformers) (3.3.0)
    Requirement already satisfied: MarkupSafe>=2.0 in /usr/local/lib/python3.10/dist-packages (from jinja2->torch>=1.11.0->sentence_transformers) (2.1.5)
    Requirement already satisfied: charset-normalizer<4,>=2 in /usr/local/lib/python3.10/dist-packages (from requests->huggingface-hub>=0.15.1->sentence_transformers) (3.3.2)
    Requirement already satisfied: idna<4,>=2.5 in /usr/local/lib/python3.10/dist-packages (from requests->huggingface-hub>=0.15.1->sentence_transformers) (3.6)
    Requirement already satisfied: urllib3<3,>=1.21.1 in /usr/local/lib/python3.10/dist-packages (from requests->huggingface-hub>=0.15.1->sentence_transformers) (2.0.7)
    Requirement already satisfied: certifi>=2017.4.17 in /usr/local/lib/python3.10/dist-packages (from requests->huggingface-hub>=0.15.1->sentence_transformers) (2024.2.2)
    Requirement already satisfied: mpmath>=0.19 in /usr/local/lib/python3.10/dist-packages (from sympy->torch>=1.11.0->sentence_transformers) (1.3.0)
    Installing collected packages: nvidia-nvtx-cu12, nvidia-nvjitlink-cu12, nvidia-nccl-cu12, nvidia-curand-cu12, nvidia-cufft-cu12, nvidia-cuda-runtime-cu12, nvidia-cuda-nvrtc-cu12, nvidia-cuda-cupti-cu12, nvidia-cublas-cu12, nvidia-cusparse-cu12, nvidia-cudnn-cu12, nvidia-cusolver-cu12, sentence_transformers
    Successfully installed nvidia-cublas-cu12-12.1.3.1 nvidia-cuda-cupti-cu12-12.1.105 nvidia-cuda-nvrtc-cu12-12.1.105 nvidia-cuda-runtime-cu12-12.1.105 nvidia-cudnn-cu12-8.9.2.26 nvidia-cufft-cu12-11.0.2.54 nvidia-curand-cu12-10.3.2.106 nvidia-cusolver-cu12-11.4.5.107 nvidia-cusparse-cu12-12.1.0.106 nvidia-nccl-cu12-2.19.3 nvidia-nvjitlink-cu12-12.4.99 nvidia-nvtx-cu12-12.1.105 sentence_transformers-2.5.1
    

# SentenceBERT 모델 로드


```python
# Bert(Bidirectional Encoder Representations from Transformers), 레이블이 지정된 작업에 대해 추가적인 미세 조정(fine-tuning)이 가능

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

웰니스 대화 스크립트 데이터셋

https://aihub.or.kr/opendata/keti-data/recognition-laguage/KETI-02-006

위 링크는 열리지 않는다 => AI hub를 참고한다.

현재는 존재하지 않으나 링크는 작용한다

이 데이터는 심리상담과 관련된 데이터셋이다. ( 강사님의 깃허브에 자료 있음 )\

https://github.com/leelang7/mental-health-chatbot/blob/master/chatbot.py


```python
df = pd.read_csv('/content/drive/MyDrive/Dataset/Chatbot/wellness_dataset_original.csv')
df.head()
```





  <div id="df-714b8435-4296-428b-9e66-42df3d09c99e" class="colab-df-container">
    <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
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
</div>
    <div class="colab-df-buttons">

  <div class="colab-df-container">
    <button class="colab-df-convert" onclick="convertToInteractive('df-714b8435-4296-428b-9e66-42df3d09c99e')"
            title="Convert this dataframe to an interactive table."
            style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px" viewBox="0 -960 960 960">
    <path d="M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z"/>
  </svg>
    </button>

  <style>
    .colab-df-container {
      display:flex;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    .colab-df-buttons div {
      margin-bottom: 4px;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

    <script>
      const buttonEl =
        document.querySelector('#df-714b8435-4296-428b-9e66-42df3d09c99e button.colab-df-convert');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      async function convertToInteractive(key) {
        const element = document.querySelector('#df-714b8435-4296-428b-9e66-42df3d09c99e');
        const dataTable =
          await google.colab.kernel.invokeFunction('convertToInteractive',
                                                    [key], {});
        if (!dataTable) return;

        const docLinkHtml = 'Like what you see? Visit the ' +
          '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
          + ' to learn more about interactive tables.';
        element.innerHTML = '';
        dataTable['output_type'] = 'display_data';
        await google.colab.output.renderOutput(dataTable, element);
        const docLink = document.createElement('div');
        docLink.innerHTML = docLinkHtml;
        element.appendChild(docLink);
      }
    </script>
  </div>


<div id="df-6b3fc749-8f08-487e-9dde-78a9073ef01f">
  <button class="colab-df-quickchart" onclick="quickchart('df-6b3fc749-8f08-487e-9dde-78a9073ef01f')"
            title="Suggest charts"
            style="display:none;">

<svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
     width="24px">
    <g>
        <path d="M19 3H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zM9 17H7v-7h2v7zm4 0h-2V7h2v10zm4 0h-2v-4h2v4z"/>
    </g>
</svg>
  </button>

<style>
  .colab-df-quickchart {
      --bg-color: #E8F0FE;
      --fill-color: #1967D2;
      --hover-bg-color: #E2EBFA;
      --hover-fill-color: #174EA6;
      --disabled-fill-color: #AAA;
      --disabled-bg-color: #DDD;
  }

  [theme=dark] .colab-df-quickchart {
      --bg-color: #3B4455;
      --fill-color: #D2E3FC;
      --hover-bg-color: #434B5C;
      --hover-fill-color: #FFFFFF;
      --disabled-bg-color: #3B4455;
      --disabled-fill-color: #666;
  }

  .colab-df-quickchart {
    background-color: var(--bg-color);
    border: none;
    border-radius: 50%;
    cursor: pointer;
    display: none;
    fill: var(--fill-color);
    height: 32px;
    padding: 0;
    width: 32px;
  }

  .colab-df-quickchart:hover {
    background-color: var(--hover-bg-color);
    box-shadow: 0 1px 2px rgba(60, 64, 67, 0.3), 0 1px 3px 1px rgba(60, 64, 67, 0.15);
    fill: var(--button-hover-fill-color);
  }

  .colab-df-quickchart-complete:disabled,
  .colab-df-quickchart-complete:disabled:hover {
    background-color: var(--disabled-bg-color);
    fill: var(--disabled-fill-color);
    box-shadow: none;
  }

  .colab-df-spinner {
    border: 2px solid var(--fill-color);
    border-color: transparent;
    border-bottom-color: var(--fill-color);
    animation:
      spin 1s steps(1) infinite;
  }

  @keyframes spin {
    0% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
      border-left-color: var(--fill-color);
    }
    20% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    30% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
      border-right-color: var(--fill-color);
    }
    40% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    60% {
      border-color: transparent;
      border-right-color: var(--fill-color);
    }
    80% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-bottom-color: var(--fill-color);
    }
    90% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
    }
  }
</style>

  <script>
    async function quickchart(key) {
      const quickchartButtonEl =
        document.querySelector('#' + key + ' button');
      quickchartButtonEl.disabled = true;  // To prevent multiple clicks.
      quickchartButtonEl.classList.add('colab-df-spinner');
      try {
        const charts = await google.colab.kernel.invokeFunction(
            'suggestCharts', [key], {});
      } catch (error) {
        console.error('Error during call to suggestCharts:', error);
      }
      quickchartButtonEl.classList.remove('colab-df-spinner');
      quickchartButtonEl.classList.add('colab-df-quickchart-complete');
    }
    (() => {
      let quickchartButtonEl =
        document.querySelector('#df-6b3fc749-8f08-487e-9dde-78a9073ef01f button');
      quickchartButtonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';
    })();
  </script>
</div>

    </div>
  </div>




# 전처리


```python
# Nan 제거
df = df.drop(columns=['Unnamed: 3'])

df.head()
```





  <div id="df-be5c457e-4e10-4776-9e8f-dc7bfc5677a2" class="colab-df-container">
    <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
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
</div>
    <div class="colab-df-buttons">

  <div class="colab-df-container">
    <button class="colab-df-convert" onclick="convertToInteractive('df-be5c457e-4e10-4776-9e8f-dc7bfc5677a2')"
            title="Convert this dataframe to an interactive table."
            style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px" viewBox="0 -960 960 960">
    <path d="M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z"/>
  </svg>
    </button>

  <style>
    .colab-df-container {
      display:flex;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    .colab-df-buttons div {
      margin-bottom: 4px;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

    <script>
      const buttonEl =
        document.querySelector('#df-be5c457e-4e10-4776-9e8f-dc7bfc5677a2 button.colab-df-convert');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      async function convertToInteractive(key) {
        const element = document.querySelector('#df-be5c457e-4e10-4776-9e8f-dc7bfc5677a2');
        const dataTable =
          await google.colab.kernel.invokeFunction('convertToInteractive',
                                                    [key], {});
        if (!dataTable) return;

        const docLinkHtml = 'Like what you see? Visit the ' +
          '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
          + ' to learn more about interactive tables.';
        element.innerHTML = '';
        dataTable['output_type'] = 'display_data';
        await google.colab.output.renderOutput(dataTable, element);
        const docLink = document.createElement('div');
        docLink.innerHTML = docLinkHtml;
        element.appendChild(docLink);
      }
    </script>
  </div>


<div id="df-6a795c58-90b7-4130-a435-75a192dc8292">
  <button class="colab-df-quickchart" onclick="quickchart('df-6a795c58-90b7-4130-a435-75a192dc8292')"
            title="Suggest charts"
            style="display:none;">

<svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
     width="24px">
    <g>
        <path d="M19 3H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zM9 17H7v-7h2v7zm4 0h-2V7h2v10zm4 0h-2v-4h2v4z"/>
    </g>
</svg>
  </button>

<style>
  .colab-df-quickchart {
      --bg-color: #E8F0FE;
      --fill-color: #1967D2;
      --hover-bg-color: #E2EBFA;
      --hover-fill-color: #174EA6;
      --disabled-fill-color: #AAA;
      --disabled-bg-color: #DDD;
  }

  [theme=dark] .colab-df-quickchart {
      --bg-color: #3B4455;
      --fill-color: #D2E3FC;
      --hover-bg-color: #434B5C;
      --hover-fill-color: #FFFFFF;
      --disabled-bg-color: #3B4455;
      --disabled-fill-color: #666;
  }

  .colab-df-quickchart {
    background-color: var(--bg-color);
    border: none;
    border-radius: 50%;
    cursor: pointer;
    display: none;
    fill: var(--fill-color);
    height: 32px;
    padding: 0;
    width: 32px;
  }

  .colab-df-quickchart:hover {
    background-color: var(--hover-bg-color);
    box-shadow: 0 1px 2px rgba(60, 64, 67, 0.3), 0 1px 3px 1px rgba(60, 64, 67, 0.15);
    fill: var(--button-hover-fill-color);
  }

  .colab-df-quickchart-complete:disabled,
  .colab-df-quickchart-complete:disabled:hover {
    background-color: var(--disabled-bg-color);
    fill: var(--disabled-fill-color);
    box-shadow: none;
  }

  .colab-df-spinner {
    border: 2px solid var(--fill-color);
    border-color: transparent;
    border-bottom-color: var(--fill-color);
    animation:
      spin 1s steps(1) infinite;
  }

  @keyframes spin {
    0% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
      border-left-color: var(--fill-color);
    }
    20% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    30% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
      border-right-color: var(--fill-color);
    }
    40% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    60% {
      border-color: transparent;
      border-right-color: var(--fill-color);
    }
    80% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-bottom-color: var(--fill-color);
    }
    90% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
    }
  }
</style>

  <script>
    async function quickchart(key) {
      const quickchartButtonEl =
        document.querySelector('#' + key + ' button');
      quickchartButtonEl.disabled = true;  // To prevent multiple clicks.
      quickchartButtonEl.classList.add('colab-df-spinner');
      try {
        const charts = await google.colab.kernel.invokeFunction(
            'suggestCharts', [key], {});
      } catch (error) {
        console.error('Error during call to suggestCharts:', error);
      }
      quickchartButtonEl.classList.remove('colab-df-spinner');
      quickchartButtonEl.classList.add('colab-df-quickchart-complete');
    }
    (() => {
      let quickchartButtonEl =
        document.querySelector('#df-6a795c58-90b7-4130-a435-75a192dc8292 button');
      quickchartButtonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';
    })();
  </script>
</div>

    </div>
  </div>





```python
df = df[~df['챗봇'].isna()]

df.head()
```





  <div id="df-d91b86a7-6f4d-4980-86f6-2d467524ff0e" class="colab-df-container">
    <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
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
</div>
    <div class="colab-df-buttons">

  <div class="colab-df-container">
    <button class="colab-df-convert" onclick="convertToInteractive('df-d91b86a7-6f4d-4980-86f6-2d467524ff0e')"
            title="Convert this dataframe to an interactive table."
            style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px" viewBox="0 -960 960 960">
    <path d="M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z"/>
  </svg>
    </button>

  <style>
    .colab-df-container {
      display:flex;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    .colab-df-buttons div {
      margin-bottom: 4px;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

    <script>
      const buttonEl =
        document.querySelector('#df-d91b86a7-6f4d-4980-86f6-2d467524ff0e button.colab-df-convert');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      async function convertToInteractive(key) {
        const element = document.querySelector('#df-d91b86a7-6f4d-4980-86f6-2d467524ff0e');
        const dataTable =
          await google.colab.kernel.invokeFunction('convertToInteractive',
                                                    [key], {});
        if (!dataTable) return;

        const docLinkHtml = 'Like what you see? Visit the ' +
          '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
          + ' to learn more about interactive tables.';
        element.innerHTML = '';
        dataTable['output_type'] = 'display_data';
        await google.colab.output.renderOutput(dataTable, element);
        const docLink = document.createElement('div');
        docLink.innerHTML = docLinkHtml;
        element.appendChild(docLink);
      }
    </script>
  </div>


<div id="df-86ccd5e8-4944-4cc6-9669-a4f9eaae1cb1">
  <button class="colab-df-quickchart" onclick="quickchart('df-86ccd5e8-4944-4cc6-9669-a4f9eaae1cb1')"
            title="Suggest charts"
            style="display:none;">

<svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
     width="24px">
    <g>
        <path d="M19 3H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zM9 17H7v-7h2v7zm4 0h-2V7h2v10zm4 0h-2v-4h2v4z"/>
    </g>
</svg>
  </button>

<style>
  .colab-df-quickchart {
      --bg-color: #E8F0FE;
      --fill-color: #1967D2;
      --hover-bg-color: #E2EBFA;
      --hover-fill-color: #174EA6;
      --disabled-fill-color: #AAA;
      --disabled-bg-color: #DDD;
  }

  [theme=dark] .colab-df-quickchart {
      --bg-color: #3B4455;
      --fill-color: #D2E3FC;
      --hover-bg-color: #434B5C;
      --hover-fill-color: #FFFFFF;
      --disabled-bg-color: #3B4455;
      --disabled-fill-color: #666;
  }

  .colab-df-quickchart {
    background-color: var(--bg-color);
    border: none;
    border-radius: 50%;
    cursor: pointer;
    display: none;
    fill: var(--fill-color);
    height: 32px;
    padding: 0;
    width: 32px;
  }

  .colab-df-quickchart:hover {
    background-color: var(--hover-bg-color);
    box-shadow: 0 1px 2px rgba(60, 64, 67, 0.3), 0 1px 3px 1px rgba(60, 64, 67, 0.15);
    fill: var(--button-hover-fill-color);
  }

  .colab-df-quickchart-complete:disabled,
  .colab-df-quickchart-complete:disabled:hover {
    background-color: var(--disabled-bg-color);
    fill: var(--disabled-fill-color);
    box-shadow: none;
  }

  .colab-df-spinner {
    border: 2px solid var(--fill-color);
    border-color: transparent;
    border-bottom-color: var(--fill-color);
    animation:
      spin 1s steps(1) infinite;
  }

  @keyframes spin {
    0% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
      border-left-color: var(--fill-color);
    }
    20% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    30% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
      border-right-color: var(--fill-color);
    }
    40% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    60% {
      border-color: transparent;
      border-right-color: var(--fill-color);
    }
    80% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-bottom-color: var(--fill-color);
    }
    90% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
    }
  }
</style>

  <script>
    async function quickchart(key) {
      const quickchartButtonEl =
        document.querySelector('#' + key + ' button');
      quickchartButtonEl.disabled = true;  // To prevent multiple clicks.
      quickchartButtonEl.classList.add('colab-df-spinner');
      try {
        const charts = await google.colab.kernel.invokeFunction(
            'suggestCharts', [key], {});
      } catch (error) {
        console.error('Error during call to suggestCharts:', error);
      }
      quickchartButtonEl.classList.remove('colab-df-spinner');
      quickchartButtonEl.classList.add('colab-df-quickchart-complete');
    }
    (() => {
      let quickchartButtonEl =
        document.querySelector('#df-86ccd5e8-4944-4cc6-9669-a4f9eaae1cb1 button');
      quickchartButtonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';
    })();
  </script>
</div>

    </div>
  </div>





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
            6.13689758e-02, -1.91311941e-01,  2.53917605e-01, -5.85018933e-01,
           -2.84426153e-01, -2.32035235e-01, -3.27080905e-01,  6.72981143e-02,
           -1.64221437e-05, -4.72336292e-01, -3.60022008e-01,  2.91879863e-01,
           -6.63860917e-01, -3.10574800e-01,  5.79524815e-01, -3.11722726e-01,
            1.47697125e-02, -2.12172404e-01,  2.22058311e-01, -1.73828721e-01,
           -3.78458530e-01, -4.20398265e-01, -2.38219023e-01,  6.38705865e-02,
           -1.15304723e-01, -2.44563967e-01, -5.00228345e-01,  1.68355703e-01,
           -6.58360064e-01, -8.91942024e-01, -6.26956999e-01, -3.21965218e-01,
           -7.05358207e-01,  3.71447384e-01, -5.45803845e-01,  7.76300654e-02,
            1.09864473e-01,  2.60926545e-01,  5.70639074e-01,  4.83376086e-01,
           -4.98233229e-01, -1.25609085e-01,  6.48291931e-02,  1.31349918e-02,
           -9.61792096e-02, -4.69374985e-01,  1.95989266e-01,  5.54263830e-01,
           -5.48072271e-02,  5.83693087e-01,  5.35227418e-01,  1.76425174e-01,
           -1.22763552e-01,  1.60846695e-01,  1.41475260e-01,  1.20705283e+00,
            6.68891788e-01, -1.77987203e-01, -2.65273869e-01, -7.37367943e-02,
           -6.52007163e-01,  7.05028951e-01,  3.38973135e-01,  7.44479477e-01,
           -2.32737795e-01,  8.69920403e-02, -1.99688762e-01,  3.89369689e-02,
            4.56739128e-01,  4.72499520e-01,  4.76455301e-01, -3.94157737e-01,
           -1.58249885e-01,  2.30056450e-01, -7.33413160e-01, -2.81495273e-01,
           -3.65628868e-01, -3.06134552e-01,  5.54837823e-01, -4.36482914e-02,
            6.42989157e-03,  6.42393470e-01, -4.23780620e-01,  3.66444916e-01,
           -2.23843262e-01, -1.02090788e+00,  2.90704876e-01, -1.98106363e-01,
            1.18598633e-01, -1.10149622e+00,  7.87631981e-03,  4.42555159e-01,
           -1.84308872e-01, -1.11383545e+00, -6.35490268e-02,  3.82617563e-01,
            3.19346376e-02,  5.33872563e-03,  1.28215075e-01,  2.67615050e-01,
           -4.09684360e-01, -2.34070778e-01,  4.23293799e-01, -1.85263842e-01,
            5.28334200e-01, -8.20845589e-02, -1.60459831e-01, -2.11379901e-01,
           -3.30953412e-02, -5.65643668e-01, -8.22234526e-02,  6.33469760e-01,
           -6.89976364e-02, -4.09638017e-01, -1.70856833e-01, -4.48212028e-02,
            2.04584464e-01, -7.15699196e-01,  2.02314153e-01,  1.36369556e-01,
           -2.02901945e-01, -6.36675537e-01,  2.00451180e-01,  3.50281745e-01,
           -6.38650358e-01, -6.02853373e-02, -1.89429522e-01, -1.09799907e-01,
           -1.02066226e-01, -3.87780815e-01, -6.51848316e-02, -1.08274007e-02,
           -1.96998224e-01, -8.77871290e-02, -6.75758243e-01, -4.23240036e-01,
            8.39257687e-02,  3.25871378e-01,  8.25180471e-01, -3.97949189e-01,
            8.68805801e-04,  5.75698256e-01,  8.25042352e-02,  9.16278139e-02,
           -5.58024883e-01, -1.46185607e-01, -7.81165481e-01,  8.25591758e-02,
           -7.54721314e-02,  1.78786382e-01,  1.14652142e-03,  2.46109247e-01,
            2.54704684e-01, -2.15057746e-01,  4.98201311e-01,  1.62567899e-01,
            3.10305148e-01,  1.72159508e-01,  6.11411750e-01,  2.40109816e-01,
            4.81477529e-01,  5.03617860e-02,  4.37822670e-01, -3.92508209e-01,
           -5.68297267e-01,  9.06163394e-01,  4.21260327e-01,  2.00669721e-01,
           -3.01369160e-01,  3.08158547e-01,  2.09018841e-01, -3.07969481e-01,
           -1.80494502e-01,  6.86860457e-02,  8.82594705e-01,  1.83178395e-01,
           -1.13062434e-01, -5.55954516e-01, -9.77990963e-03, -1.27596647e-01,
           -4.29738820e-01, -1.13016844e-01, -1.89255252e-01,  2.49697596e-01,
           -2.61328191e-01, -3.91260356e-01, -2.63828367e-01,  1.65253907e-01,
            5.30026555e-01, -3.99859995e-01, -3.08907837e-01,  3.59966516e-01,
            4.66616899e-01, -3.19502912e-02, -5.92679322e-01, -3.42451990e-01,
           -1.90145835e-01, -3.22044581e-01, -2.08034188e-01, -4.68787730e-01,
            2.60370046e-01, -3.10487688e-01,  3.08493614e-01, -5.58807552e-01,
           -3.15932244e-01, -6.11169219e-01, -8.03783350e-03, -3.57464164e-01,
           -7.77107403e-02, -1.36108741e-01,  3.62246543e-01, -1.04268157e+00,
            2.66694814e-01, -3.39090973e-01,  6.24717772e-03,  5.45241572e-02,
           -1.02638483e-01, -2.67193288e-01, -5.36906838e-01, -3.32568318e-01,
            1.79672986e-01, -4.72883806e-02,  2.58301169e-01,  6.13248684e-02,
            4.89077903e-02, -2.22134590e-01, -5.60372889e-01,  2.79794514e-01,
           -2.88628787e-01,  3.76666129e-01, -6.29106820e-01, -5.76252937e-01,
            5.28727233e-01, -5.10725677e-01, -1.89374119e-01, -3.58017951e-01,
           -5.50130785e-01,  1.71586558e-01,  9.91874337e-02, -1.48076639e-01,
            9.65982452e-02,  2.78743267e-01,  1.04141104e+00, -2.29346737e-01,
            5.47820151e-01, -1.33991152e-01,  4.53759044e-01, -1.24763198e-01,
           -3.36636394e-01,  1.19638935e-01, -2.03772988e-02, -4.97905403e-01,
           -3.44481498e-01,  5.91197968e-01,  2.08580270e-01, -4.75013077e-01,
            1.56643897e-01, -2.28076190e-01,  1.56071216e-01,  7.77589202e-01,
            5.55079877e-02,  4.68444973e-01, -4.18879166e-02, -2.33576134e-01,
            6.30587161e-01,  3.43996465e-01, -3.72299761e-01,  4.22593117e-01,
           -1.28934932e+00,  7.28118241e-01,  7.39577830e-01, -1.57391548e-01,
           -2.73995489e-01,  3.68654132e-02, -5.34251750e-01,  5.79056621e-01,
            4.52849448e-01, -1.12652934e+00,  8.49229321e-02,  4.00368758e-02,
            5.59743702e-01, -1.44835666e-01, -1.36150554e-01,  2.27955148e-01,
           -2.42799625e-01,  4.50197995e-01, -4.44726527e-01,  5.72470203e-02,
           -3.26299220e-01,  3.96916345e-02, -3.63902330e-01, -7.14616120e-01,
            2.71114297e-02,  3.17301691e-01,  2.74653703e-01, -2.76966602e-01,
           -6.03288233e-01, -1.80380326e-02, -1.60021320e-01,  4.97588426e-01,
           -4.02046502e-01, -3.95431280e-01, -1.88366696e-01,  2.06625089e-01,
            3.04095596e-01, -3.11066568e-01,  3.70665714e-02, -9.00971949e-01,
            2.54859865e-01, -2.00540498e-01,  1.62926331e-01, -3.99485856e-01,
            4.31148916e-01,  9.35604751e-01,  7.70608962e-01, -1.80281132e-01,
            7.98517704e-01, -5.61508060e-01,  3.77493650e-01, -3.04625213e-01,
            3.04160535e-01, -7.78200626e-02, -1.80557698e-01,  1.92049056e-01,
            7.25668132e-01, -1.46759525e-01,  2.77359724e-01, -3.10689747e-01,
            3.84764463e-01, -9.89015773e-02,  1.33970501e-02, -3.74624312e-01,
            6.19450271e-01, -1.19058944e-01,  5.74723661e-01,  4.28427219e-01,
            1.73577100e-01, -5.50395608e-01,  3.84832591e-01, -1.56954601e-01,
           -1.65572762e-02,  2.90999711e-02, -1.59130126e-01,  7.94794083e-01,
            4.56630558e-01,  1.08082496e-01,  1.70058489e-01,  3.50349173e-02,
           -2.01882914e-01, -1.83400080e-01,  2.56279379e-01,  8.36629808e-01,
            5.35587013e-01,  4.41516191e-01,  3.18022557e-02, -2.46380255e-01,
           -2.17682049e-01,  2.26683900e-01, -2.78985828e-01,  4.22985069e-02,
           -2.80858815e-01,  2.65039027e-01, -3.63108903e-01, -6.51625335e-01,
            1.56545967e-01, -1.82188060e-02, -2.57317454e-01,  1.89768746e-01,
           -1.36993766e-01,  4.79500070e-02,  4.41078663e-01, -2.83327430e-01,
            1.25181943e-01,  4.78675276e-01,  3.22104752e-01, -3.69765490e-01,
           -6.76260352e-01,  3.52737606e-01,  1.97328731e-01,  3.01312447e-01,
           -1.70432106e-01,  1.91599037e-02, -4.52054977e-01, -2.89518893e-01,
            6.21264160e-01,  7.11012244e-01,  7.38060474e-02, -1.13718852e-01,
            4.01600152e-01,  3.53704661e-01, -1.70214266e-01, -6.32789314e-01,
           -5.92871666e-01,  4.90854591e-01,  3.11830819e-01, -1.06595680e-01,
            6.99721217e-01, -6.40676022e-01,  9.11572352e-02,  1.48974419e-01,
            1.12393862e-02,  1.35593802e-01,  1.30469859e-01, -3.76635551e-01,
            5.62429786e-01, -4.00536329e-01,  2.05565050e-01, -2.96904653e-01,
           -5.38131595e-01, -4.20320749e-01, -6.31194189e-02,  1.50621459e-01,
           -3.38527828e-01,  5.21204472e-01, -4.65558946e-01, -3.12167436e-01,
           -3.87227327e-01, -4.87661809e-01, -1.49875879e-03,  6.48479939e-01,
            1.56931266e-01, -3.11310470e-01, -8.74779895e-02, -2.41719827e-01,
           -8.19946826e-01, -9.02200878e-01, -8.39587510e-01,  2.95948144e-02,
            3.79318893e-01, -3.63524646e-01, -2.12089062e-01,  3.26290220e-01,
           -5.83360732e-01, -2.70313531e-01, -1.90380454e-01,  5.40128171e-01,
            4.60663587e-01,  2.41606340e-01, -1.55656427e-01,  1.10515499e+00,
            3.81636232e-01, -1.74268335e-01,  1.18968032e-01, -2.14279577e-01,
            3.82164270e-01, -3.88768673e-01,  5.05674720e-01,  1.41618282e-01,
            3.24829906e-01,  1.17207609e-01,  1.90372676e-01, -1.00517496e-01,
            7.24769384e-02, -8.40552330e-01, -1.66888610e-01,  1.35378689e-01,
           -2.34575734e-01, -6.08667061e-02, -2.85461277e-01,  6.98202968e-01,
           -3.75756890e-01, -1.00717820e-01, -6.50324464e-01,  1.71460599e-01,
           -4.01935756e-01,  5.49870372e-01,  8.29208121e-02, -1.16668515e-01,
            7.88919106e-02,  8.92655998e-02,  4.86943007e-01, -4.59339142e-01,
            9.24550444e-02,  7.74140537e-01, -9.86551821e-01, -1.73161298e-01,
           -1.85814455e-01, -5.37258089e-01, -2.54959136e-01, -4.34431612e-01,
            3.07515357e-02, -8.44853997e-01, -2.82899350e-01, -1.30699563e+00,
           -6.73802793e-01, -6.54637635e-01,  4.86585289e-01, -2.13963747e-01,
            5.78385353e-01,  2.93961704e-01, -4.04392332e-01, -7.01608121e-01,
           -1.73342809e-01,  3.51329058e-01,  1.19921379e-01,  6.68857843e-02,
           -1.38902619e-01, -5.46096921e-01,  1.97514266e-01,  8.67951140e-02,
           -7.13970482e-01,  1.36450931e-01,  4.15191829e-01, -1.77055016e-01,
           -2.05800578e-01,  6.08606398e-01, -1.40279219e-01,  2.65653640e-01,
            5.22155762e-02, -1.10528894e-01,  3.22062284e-01, -2.36262888e-01,
            2.32510462e-01, -4.01646525e-01,  9.92268398e-02,  2.12363809e-01,
            4.54302639e-01,  1.93777278e-01,  1.90577134e-01,  6.74432576e-01,
            6.50040656e-02,  1.11985549e-01, -6.33725375e-02,  4.36560661e-01,
            7.08307683e-01,  4.61303800e-01, -4.12172556e-01, -3.44057798e-01,
           -3.16411763e-01, -3.95597935e-01, -1.39263440e-02, -2.46823683e-01,
           -5.97274780e-01, -4.93520200e-01,  2.20008828e-02, -6.79914653e-01,
           -1.32439723e-02, -1.62386179e-01, -4.11045879e-01, -4.95454729e-01,
           -2.55482137e-01, -1.97103217e-01, -3.67735237e-01, -6.45768642e-01,
           -1.24110138e+00,  6.32802606e-01,  1.81258067e-01,  1.86036795e-01,
           -2.75285721e-01,  4.46098112e-02, -3.82379383e-01,  6.66201860e-03,
           -2.56120652e-01,  6.45592362e-02,  3.85579914e-01, -2.17116490e-01,
           -9.55902755e-01, -2.15195730e-01,  4.17431086e-01,  2.25791305e-01,
           -4.59714144e-01, -6.38202012e-01, -5.95626473e-01, -8.46446157e-02,
           -5.84241450e-01, -5.22198141e-01,  6.69408664e-02,  5.54487526e-01,
            2.70057648e-01,  1.14965916e-01,  5.91447711e-01, -9.82941985e-02,
           -8.05462420e-01,  1.21461146e-01,  3.13535213e-01, -1.09967031e-01,
            4.52007383e-01,  1.56954378e-01, -5.73436201e-01, -4.65392888e-01,
           -5.09206392e-02,  1.38996601e-01, -1.79907665e-01,  3.46671343e-02,
            3.28499317e-01, -3.08243096e-01, -3.20267886e-01, -7.92234957e-01,
           -5.56746185e-01,  2.58380502e-01, -1.15094580e-01, -9.68656912e-02,
           -3.73692989e-01,  4.09340650e-01, -2.20842004e-01, -7.74916470e-01,
           -3.03384870e-01,  6.75034583e-01, -5.39470494e-01, -1.12011433e-01,
           -4.62304384e-01,  4.86268222e-01,  3.34404618e-01,  3.11685115e-01,
            1.37020037e-01,  1.20470092e-01, -7.88884461e-01,  2.49228209e-01,
            3.69006127e-01, -3.55894528e-02,  9.60911870e-01,  4.11124021e-01,
            3.08286756e-01, -8.89444873e-02,  5.68126321e-01, -2.19461650e-01,
            6.64631426e-01,  1.33868188e-01, -9.25525278e-02, -4.02042389e-01,
           -2.44861498e-01,  4.75438178e-01, -5.59731126e-01,  4.23794597e-01,
            7.64948800e-02, -5.34291625e-01,  9.61298719e-02, -1.98235782e-03,
            1.52365491e-01,  2.72849537e-02, -2.30466038e-01, -6.57617092e-01,
           -8.41355026e-01,  6.93241954e-01, -1.15385902e+00,  3.39940459e-01,
            6.39491558e-01, -3.07582058e-02, -9.44079310e-02, -3.81875187e-01,
           -1.51736423e-01,  3.89264524e-01,  6.98378742e-01, -3.58799458e-01,
            1.73822060e-01, -7.02525228e-02,  2.98745722e-01,  4.25366193e-01,
            3.03247511e-01,  4.43037711e-02, -1.19768485e-01,  1.56500697e-01,
            5.65420032e-01, -6.15675092e-01, -4.82268035e-01, -6.78222924e-02,
           -2.88993359e-01, -4.48901862e-01, -5.26656330e-01,  3.26252192e-01,
            5.28493106e-01,  1.38639897e-01, -7.72885025e-01, -5.26200950e-01,
            3.72999191e-01,  3.05722237e-01,  6.69044316e-01, -5.61777391e-02,
            4.14461136e-01,  2.05879182e-01,  5.97795546e-01,  1.40301362e-01,
           -3.20007913e-02,  3.41693938e-01,  4.73953821e-02, -1.09114289e-01,
           -4.06928003e-01, -1.22424997e-01, -1.50909796e-01,  2.56388724e-01,
            1.39330504e-02, -2.18360871e-01,  4.44866210e-01,  5.06846905e-01,
           -2.75422782e-01,  8.68547559e-01, -1.61321908e-01, -8.95696878e-03,
            2.11986497e-01,  1.03369892e-01, -1.06486984e-01, -3.07435185e-01,
            6.36290729e-01,  4.12892923e-02,  3.60755846e-02, -4.70254607e-02,
           -1.08839154e-01, -8.88763461e-03, -1.25093892e-01, -5.34434080e-01,
           -4.71274167e-01,  1.96804956e-01,  3.41911651e-02, -5.81325233e-01,
            1.44223953e-02,  5.15582442e-01,  1.34000644e-01, -1.61631599e-01,
           -1.97617546e-01, -6.94736481e-01, -5.58743775e-01,  9.49026644e-01,
           -1.94514066e-01,  4.25471328e-02,  4.92083937e-01, -5.84431648e-01,
            6.24192119e-01,  4.92841095e-01,  1.22476719e-01, -4.79154140e-01,
            5.99286914e-01,  3.32147151e-01,  1.98169842e-01,  5.98622821e-02,
           -6.37879968e-03, -1.07340567e-01, -1.58823594e-01, -1.40380012e-02,
           -3.10648024e-01,  4.95430499e-01, -1.67429179e-01, -6.59497231e-02,
           -1.76547453e-01, -4.49961156e-01, -2.71651357e-01, -1.50339961e-01,
           -1.87571868e-01, -5.60149968e-01,  1.33678943e-01, -4.36347544e-01,
            1.28170326e-01, -1.43580198e-01,  2.30380893e-01, -5.46675883e-02,
            3.71279776e-01,  1.98934287e-01,  4.64870483e-01,  3.64101559e-01,
           -4.83271256e-02,  2.67422974e-01, -7.10749209e-01,  5.07541671e-02],
          dtype=float32)



# 유저 대화내용 인코딩

유저의 모든 발화를 임베딩에 저장한다.


```python
df['embedding'] = pd.Series([[]] * len(df)) # dummy

df['embedding'] = df['유저'].map(lambda x: list(model.encode(x)))

df.head()
```





  <div id="df-896f4116-352c-41a2-a438-c7b328549261" class="colab-df-container">
    <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
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
</div>
    <div class="colab-df-buttons">

  <div class="colab-df-container">
    <button class="colab-df-convert" onclick="convertToInteractive('df-896f4116-352c-41a2-a438-c7b328549261')"
            title="Convert this dataframe to an interactive table."
            style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px" viewBox="0 -960 960 960">
    <path d="M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z"/>
  </svg>
    </button>

  <style>
    .colab-df-container {
      display:flex;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    .colab-df-buttons div {
      margin-bottom: 4px;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

    <script>
      const buttonEl =
        document.querySelector('#df-896f4116-352c-41a2-a438-c7b328549261 button.colab-df-convert');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      async function convertToInteractive(key) {
        const element = document.querySelector('#df-896f4116-352c-41a2-a438-c7b328549261');
        const dataTable =
          await google.colab.kernel.invokeFunction('convertToInteractive',
                                                    [key], {});
        if (!dataTable) return;

        const docLinkHtml = 'Like what you see? Visit the ' +
          '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
          + ' to learn more about interactive tables.';
        element.innerHTML = '';
        dataTable['output_type'] = 'display_data';
        await google.colab.output.renderOutput(dataTable, element);
        const docLink = document.createElement('div');
        docLink.innerHTML = docLinkHtml;
        element.appendChild(docLink);
      }
    </script>
  </div>


<div id="df-2786bbaa-267c-49d5-8970-36824c0ba3c4">
  <button class="colab-df-quickchart" onclick="quickchart('df-2786bbaa-267c-49d5-8970-36824c0ba3c4')"
            title="Suggest charts"
            style="display:none;">

<svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
     width="24px">
    <g>
        <path d="M19 3H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zM9 17H7v-7h2v7zm4 0h-2V7h2v10zm4 0h-2v-4h2v4z"/>
    </g>
</svg>
  </button>

<style>
  .colab-df-quickchart {
      --bg-color: #E8F0FE;
      --fill-color: #1967D2;
      --hover-bg-color: #E2EBFA;
      --hover-fill-color: #174EA6;
      --disabled-fill-color: #AAA;
      --disabled-bg-color: #DDD;
  }

  [theme=dark] .colab-df-quickchart {
      --bg-color: #3B4455;
      --fill-color: #D2E3FC;
      --hover-bg-color: #434B5C;
      --hover-fill-color: #FFFFFF;
      --disabled-bg-color: #3B4455;
      --disabled-fill-color: #666;
  }

  .colab-df-quickchart {
    background-color: var(--bg-color);
    border: none;
    border-radius: 50%;
    cursor: pointer;
    display: none;
    fill: var(--fill-color);
    height: 32px;
    padding: 0;
    width: 32px;
  }

  .colab-df-quickchart:hover {
    background-color: var(--hover-bg-color);
    box-shadow: 0 1px 2px rgba(60, 64, 67, 0.3), 0 1px 3px 1px rgba(60, 64, 67, 0.15);
    fill: var(--button-hover-fill-color);
  }

  .colab-df-quickchart-complete:disabled,
  .colab-df-quickchart-complete:disabled:hover {
    background-color: var(--disabled-bg-color);
    fill: var(--disabled-fill-color);
    box-shadow: none;
  }

  .colab-df-spinner {
    border: 2px solid var(--fill-color);
    border-color: transparent;
    border-bottom-color: var(--fill-color);
    animation:
      spin 1s steps(1) infinite;
  }

  @keyframes spin {
    0% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
      border-left-color: var(--fill-color);
    }
    20% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    30% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
      border-right-color: var(--fill-color);
    }
    40% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    60% {
      border-color: transparent;
      border-right-color: var(--fill-color);
    }
    80% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-bottom-color: var(--fill-color);
    }
    90% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
    }
  }
</style>

  <script>
    async function quickchart(key) {
      const quickchartButtonEl =
        document.querySelector('#' + key + ' button');
      quickchartButtonEl.disabled = true;  // To prevent multiple clicks.
      quickchartButtonEl.classList.add('colab-df-spinner');
      try {
        const charts = await google.colab.kernel.invokeFunction(
            'suggestCharts', [key], {});
      } catch (error) {
        console.error('Error during call to suggestCharts:', error);
      }
      quickchartButtonEl.classList.remove('colab-df-spinner');
      quickchartButtonEl.classList.add('colab-df-quickchart-complete');
    }
    (() => {
      let quickchartButtonEl =
        document.querySelector('#df-2786bbaa-267c-49d5-8970-36824c0ba3c4 button');
      quickchartButtonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';
    })();
  </script>
</div>

    </div>
  </div>





```python
# 데이터프레임을 CSV 파일로 저장

df.to_csv('wellness_dataset.csv', index=False)
```

# 간단한 챗봇


```python
text = '요즘 머리가 아프고 너무 힘들어' # 유저가 이러한 문장을 쳤을 때
embedding = model.encode(text)
```


```python
# 유저가 쓴 문장과 유사한 문장 찾기, 코사인 유사도

df['distance'] = df['embedding'].map(lambda x: cosine_similarity([embedding], [x]).squeeze()) # 코싸인 유사도가 높은 것 찾아서 어떤 유저가 얘기했을 때 임베딩한 가장 가까운 결과와의 거리를 구한 후 distance에 넣어준다.

# 즉, distance가 1에 가장 가까운 문장이 출력되는 구조이다.
df.head()
```





  <div id="df-4892541c-8197-42bd-8b9e-1a77d80591ff" class="colab-df-container">
    <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
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
</div>
    <div class="colab-df-buttons">

  <div class="colab-df-container">
    <button class="colab-df-convert" onclick="convertToInteractive('df-4892541c-8197-42bd-8b9e-1a77d80591ff')"
            title="Convert this dataframe to an interactive table."
            style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px" viewBox="0 -960 960 960">
    <path d="M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z"/>
  </svg>
    </button>

  <style>
    .colab-df-container {
      display:flex;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    .colab-df-buttons div {
      margin-bottom: 4px;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

    <script>
      const buttonEl =
        document.querySelector('#df-4892541c-8197-42bd-8b9e-1a77d80591ff button.colab-df-convert');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      async function convertToInteractive(key) {
        const element = document.querySelector('#df-4892541c-8197-42bd-8b9e-1a77d80591ff');
        const dataTable =
          await google.colab.kernel.invokeFunction('convertToInteractive',
                                                    [key], {});
        if (!dataTable) return;

        const docLinkHtml = 'Like what you see? Visit the ' +
          '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
          + ' to learn more about interactive tables.';
        element.innerHTML = '';
        dataTable['output_type'] = 'display_data';
        await google.colab.output.renderOutput(dataTable, element);
        const docLink = document.createElement('div');
        docLink.innerHTML = docLinkHtml;
        element.appendChild(docLink);
      }
    </script>
  </div>


<div id="df-9341c760-7ed7-45fa-b001-5b96ca19ea61">
  <button class="colab-df-quickchart" onclick="quickchart('df-9341c760-7ed7-45fa-b001-5b96ca19ea61')"
            title="Suggest charts"
            style="display:none;">

<svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
     width="24px">
    <g>
        <path d="M19 3H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zM9 17H7v-7h2v7zm4 0h-2V7h2v10zm4 0h-2v-4h2v4z"/>
    </g>
</svg>
  </button>

<style>
  .colab-df-quickchart {
      --bg-color: #E8F0FE;
      --fill-color: #1967D2;
      --hover-bg-color: #E2EBFA;
      --hover-fill-color: #174EA6;
      --disabled-fill-color: #AAA;
      --disabled-bg-color: #DDD;
  }

  [theme=dark] .colab-df-quickchart {
      --bg-color: #3B4455;
      --fill-color: #D2E3FC;
      --hover-bg-color: #434B5C;
      --hover-fill-color: #FFFFFF;
      --disabled-bg-color: #3B4455;
      --disabled-fill-color: #666;
  }

  .colab-df-quickchart {
    background-color: var(--bg-color);
    border: none;
    border-radius: 50%;
    cursor: pointer;
    display: none;
    fill: var(--fill-color);
    height: 32px;
    padding: 0;
    width: 32px;
  }

  .colab-df-quickchart:hover {
    background-color: var(--hover-bg-color);
    box-shadow: 0 1px 2px rgba(60, 64, 67, 0.3), 0 1px 3px 1px rgba(60, 64, 67, 0.15);
    fill: var(--button-hover-fill-color);
  }

  .colab-df-quickchart-complete:disabled,
  .colab-df-quickchart-complete:disabled:hover {
    background-color: var(--disabled-bg-color);
    fill: var(--disabled-fill-color);
    box-shadow: none;
  }

  .colab-df-spinner {
    border: 2px solid var(--fill-color);
    border-color: transparent;
    border-bottom-color: var(--fill-color);
    animation:
      spin 1s steps(1) infinite;
  }

  @keyframes spin {
    0% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
      border-left-color: var(--fill-color);
    }
    20% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    30% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
      border-right-color: var(--fill-color);
    }
    40% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    60% {
      border-color: transparent;
      border-right-color: var(--fill-color);
    }
    80% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-bottom-color: var(--fill-color);
    }
    90% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
    }
  }
</style>

  <script>
    async function quickchart(key) {
      const quickchartButtonEl =
        document.querySelector('#' + key + ' button');
      quickchartButtonEl.disabled = true;  // To prevent multiple clicks.
      quickchartButtonEl.classList.add('colab-df-spinner');
      try {
        const charts = await google.colab.kernel.invokeFunction(
            'suggestCharts', [key], {});
      } catch (error) {
        console.error('Error during call to suggestCharts:', error);
      }
      quickchartButtonEl.classList.remove('colab-df-spinner');
      quickchartButtonEl.classList.add('colab-df-quickchart-complete');
    }
    (() => {
      let quickchartButtonEl =
        document.querySelector('#df-9341c760-7ed7-45fa-b001-5b96ca19ea61 button');
      quickchartButtonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';
    })();
  </script>
</div>

    </div>
  </div>





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
    
