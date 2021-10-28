# 함수

# 모델 생성

## 입력값 : 토큰화된 통합 데이터
## 출력값 : TF-IDF 가중치 기반 LDA 모델 / 토픽 개수:10, 최소 빈도 단어 : 5, 최빈도 단어 : 30 

def create_model(all_itmes):

    model = tp.LDAModel(k=10, alpha=0.1, eta=0.01, min_cf=5, rm_top=30, tw=tp.TermWeight.IDF)
    # LDAModel을 생성
    # 토픽의 개수(k)는 10개, alpha 파라미터는 0.1, eta 파라미터는 0.01
    # 전체 말뭉치에 5회 미만 등장한 단어들은 제거

    for row in all_items.content:
        model.add_doc(row) # 행 별로 model에 추가합니다.

    # 학습 준비 상태 확인
    model.train(0) 
    print('Total docs:', len(model.docs))
    print('Total words:', model.num_words)
    print('Vocab size:', model.num_vocabs)

    # 다음 구문은 train을 총 200회 반복하면서, 
    # 매 단계별로 로그 가능도 값을 출력해줍니다.
    # 혹은 단순히 model.train(200)으로 200회 반복도 가능합니다.
    for i in range(200):
        model.train(1)

    return model.save('TF-IDF_model.bin')