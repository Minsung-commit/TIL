# Word2Vec embedding

# Word2Vec embedding
from gensim.models import Word2Vec
from numpy import dot
from numpy.linalg import norm
import numpy as np

# GPU = True
# if GPU: # GPU
#     import cupy as np
#     np.cuda.set_allocator(np.cuda.MemoryPool().malloc)
#     #np.add.at = np.scatter_add

#     print('\033[92m' + '-' * 60 + '\033[0m')
#     print(' ' * 23 + '\033[92mGPU Mode (cupy)\033[0m')
#     print('\033[92m' + '-' * 60 + '\033[0m\n')
# else :
#     import numpy as np


### 문장 벡터
def embedding(data):
    Word2Vec(i_data.content, size=100, window = 2, min_count=3, workers=4, iter=100, sg=1)
    
    return embedding_model

def get_sentence_mean_vector(morphs):
    vector = []
    for i in morphs:
        try:
            vector.append(embedding_model.wv[i])
        except KeyError as e:
            pass
    try:
        return sum(vector)/len(vector)
    except IndexError as e:
        pass

# 코사인 유사도
def cos_sim(A, B):
    return dot(A, B)/(norm(A)*norm(B))

# 추천 알고리즘
def creation_matrix(i_data):

    embedding_model = Word2Vec(i_data.content, size=100, window = 2, min_count=3, workers=4, iter=100, sg=1)

    i_data['wv'] = i_data['content'].map(get_sentence_mean_vector) # 100차원 벡터값 생성

    ## 코사인 유사도 행렬 생성

    wv_matrix = np.asarray(i_data.wv)

    rows = []
    matrix = []
    for i in range(608):
        for x in range(608):
            cos_sim = dot(wv_matrix[i], wv_matrix[x])/(norm(wv_matrix[i])*norm(wv_matrix[x]))
            rows.append(cos_sim)
        matrix.append(rows)
    rows=[]

    matrix = np.array(matrix)

    return matrix




def insta_REC(name, cosine_sim=matrix):
    ##인덱스 테이블 만들기##

    indices = pd.Series(i_data.index, index=i_data.name).drop_duplicates()

    #입력한 숙소로부터 인덱스 가져오기
    idx = indices[name]

    # 모든 숙소에 대해서 해당 숙소와의 유사도를 구하기
    sim_scores = list(enumerate(cosine_sim[idx]))

    # 유사도에 따라 숙소들을 정렬
    sim_scores = sorted(sim_scores, key=lambda x:x[1], reverse = True)

    # 가장 유사한 10개의 숙소를 받아옴
    sim_scores = sim_scores[1:11]

    # 가장 유사한 10개 숙소의 인덱스 받아옴
    insta_indices = [i[0] for i in sim_scores]
        
    #기존에 읽어들인 데이터에서 해당 인덱스의 값들을 가져온다. 그리고 스코어 열을 추가하여 코사인 유사도도 확인할 수 있게 한다.
    result_data = i_data.iloc[insta_indices].copy()
    result_data['score'] = [i[1] for i in sim_scores]
        
    # 읽어들인 데이터에서 콘텐츠 부분만 제거, 제목과 스코어만 보이게 함
    # del result_data['content']
    del result_data['wv']
    # del result_data['token_nolist']

    # 가장 유사한 10개의 숙소의 제목을 리턴
    return result_data


