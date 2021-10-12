## 인스타그램 데이터 전처리
# 입력값 : 전처리 되기 전 인스타 크롤링데이터(csv파일 형식)
# 출력값 : 전처리 된 인스타 데이터


def insta_preporcessing(insta):

    # 데이터 로드
    insta_data = pd.read_csv('insta')

    # 1차 클린징(영어, 특수문자, 숫자제거)
    for i in range(len(insta_data.content)):
        insta_data.content[i] = pp.sub_special(insta_data.content[i])

    # 토큰화
    for i in range(len(insta_data.content)):
        insta_data.content[i]= pp.tokenize(insta_data.content[i])

    # 2차 클린징(불용어 처리)
        pp.word_cleansing(insta_data)

    return insta_data
